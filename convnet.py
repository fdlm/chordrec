from __future__ import print_function
import os
import collections
import theano
import theano.tensor as tt
import lasagne as lnn
import yaml
from sacred import Experiment
from operator import itemgetter


import nn
import dmgr
from nn.utils import Colors

import test
import data
import targets
import features
import dnn
from exp_utils import PickleAndSymlinkObserver, TempDir, create_optimiser

# Initialise Sacred experiment
ex = Experiment('Convolutional Neural Network')
ex.observers.append(PickleAndSymlinkObserver())
data.add_sacred_config(ex)
features.add_sacred_config(ex)
targets.add_sacred_config(ex)


def compute_loss(prediction, target):
    # need to clip predictions for numerical stability
    eps = 1e-7
    pred_clip = tt.clip(prediction, eps, 1.-eps)
    return lnn.objectives.categorical_crossentropy(pred_clip, target).mean()


def stack_layers(net, batch_norm, convs):

    for i, conv in enumerate(convs):
        if not conv:
            continue
        for k in range(conv['num_layers']):
            net = lnn.layers.Conv2DLayer(
                net, num_filters=conv['num_filters'],
                filter_size=conv['filter_size'],
                nonlinearity=lnn.nonlinearities.rectify,
                name='Conv_{}_{}'.format(i, k))
            if batch_norm:
                net = lnn.layers.batch_norm(net)

        net = lnn.layers.MaxPool2DLayer(net, pool_size=conv['pool_size'],
                                        name='Pool_{}'.format(i))
        net = lnn.layers.DropoutLayer(net, p=conv['dropout'])

    return net


def add_gap_out(net, gap, batch_norm, out_size):
    net = lnn.layers.Conv2DLayer(
        net, num_filters=gap['num_filters'], filter_size=gap['filter_size'],
        pad=0, nonlinearity=lnn.nonlinearities.rectify,
        name='Gap_Filters')
    if batch_norm:
        net = lnn.layers.batch_norm(net)
    net = lnn.layers.DropoutLayer(net, p=gap['dropout'])

    net = lnn.layers.Conv2DLayer(
        net, num_filters=gap['num_filters'], filter_size=1,
        pad=0, nonlinearity=lnn.nonlinearities.rectify,
        name='Gap_Filters_Single')
    if batch_norm:
        net = lnn.layers.batch_norm(net)
    net = lnn.layers.DropoutLayer(net, p=gap['dropout'])

    # output classification layer
    net = lnn.layers.Conv2DLayer(
        net, num_filters=out_size, filter_size=1,
        nonlinearity=lnn.nonlinearities.rectify, name='Output_Conv')
    if batch_norm:
        net = lnn.layers.batch_norm(net)

    net = lnn.layers.Pool2DLayer(
        net, pool_size=net.output_shape[-2:], ignore_border=False,
        mode='average_exc_pad', name='GlobalAveragePool')
    net = lnn.layers.FlattenLayer(net, name='Flatten')
    net = lnn.layers.NonlinearityLayer(
        net, nonlinearity=lnn.nonlinearities.softmax, name='output')

    return net


def build_net(feature_shape, batch_size, optimiser, out_size,
              batch_norm, conv1, conv2, conv3, dense, global_avg_pool, l2, l1):

    # input variables
    feature_var = tt.tensor3('feature_input', dtype='float32')
    target_var = tt.matrix('target_output', dtype='float32')

    # stack more layers
    net = lnn.layers.InputLayer(name='input',
                                shape=(batch_size,) + feature_shape,
                                input_var=feature_var)

    # reshape to 1 "color" channel
    net = lnn.layers.reshape(net, shape=(-1, 1) + feature_shape,
                             name='reshape')

    net = stack_layers(net, batch_norm, [conv1, conv2, conv3])

    if dense:
        net = dnn.stack_layers(net, **dense)
        # output classification layer
        net = lnn.layers.DenseLayer(net, name='output', num_units=out_size,
                                    nonlinearity=lnn.nonlinearities.softmax)
    elif global_avg_pool:
        net = add_gap_out(net, global_avg_pool, batch_norm, out_size)
    else:
        raise RuntimeError('Need to specify output architecture!')

    # create train function
    prediction = lnn.layers.get_output(net)
    l2_penalty = lnn.regularization.regularize_network_params(
            net, lnn.regularization.l2) * l2
    l1_penalty = lnn.regularization.regularize_network_params(
        net, lnn.regularization.l1) * l1
    loss = compute_loss(prediction, target_var) + l2_penalty + l1_penalty
    params = lnn.layers.get_all_params(net, trainable=True)
    updates = optimiser(loss, params)
    train = theano.function([feature_var, target_var], loss,
                           updates=updates)

    # create test and process function. process just computes the prediction
    # without computing the loss, and thus does not need target labels
    test_prediction = lnn.layers.get_output(net, deterministic=True)
    test_loss = compute_loss(test_prediction, target_var) + l2_penalty
    test = theano.function([feature_var, target_var],
                           [test_loss, test_prediction])
    process = theano.function([feature_var], test_prediction)

    return nn.NeuralNetwork(net, train, test, process)


@ex.config
def config():
    observations = 'results'

    datasource = dict(
            context_size=7,
    )

    feature_extractor = None

    target = None

    net = dict(
        batch_norm=False,
        conv1=dict(
            num_layers=2,
            num_filters=32,
            filter_size=(3, 3),
            pool_size=(1, 2),
            dropout=0.5,
        ),
        conv2=dict(
            num_layers=1,
            num_filters=64,
            filter_size=(3, 3),
            pool_size=(1, 2),
            dropout=0.5,
        ),
        conv3={},
        dense=dict(
            num_layers=1,
            num_units=512,
            dropout=0.5,
            nonlinearity='rectify',
            batch_norm=False
        ),
        global_avg_pool=None,
        l2=1e-4,
        l1=0
    )

    optimiser = dict(
        name='adam',
        params=dict(
                learning_rate=0.001
        ),
        schedule=None
    )

    training = dict(
        num_epochs=500,
        early_stop=20,
        early_stop_acc=True,
        batch_size=512,
    )

    testing = dict(
        test_on_val=False
    )


@ex.named_config
def third_conv_layer():
    net = dict(
        conv3=dict(
            num_layers=1,
            num_filters=64,
            filter_size=(3, 3),
            pool_size=(1, 2),
            dropout=0.5,
        )
    )


@ex.named_config
def gap_classifier():
    net = dict(
        dense=None,
        global_avg_pool=dict(
            num_filters=512,
            filter_size=(3, 3),
            dropout=0.5
        )
    )


@ex.named_config
def learn_rate_schedule():
    optimiser = dict(
        schedule=dict(
            interval=10,
            factor=0.5
        )
    )


@ex.automain
def main(_config, _run, observations, datasource, net, feature_extractor,
         target, optimiser, training, testing):

    if feature_extractor is None:
        print(Colors.red('ERROR: Specify a feature extractor!'))
        return 1

    # Load data sets
    print(Colors.red('Loading data...\n'))

    target_computer = targets.create_target(
        feature_extractor['params']['fps'],
        target
    )

    if not isinstance(datasource['test_fold'], collections.Iterable):
        datasource['test_fold'] = [datasource['test_fold']]

    if not isinstance(datasource['val_fold'], collections.Iterable):
        datasource['val_fold'] = [datasource['val_fold']]

        # if no validation folds are specified, always use the
        # 'None' and determine validation fold automatically
        if datasource['val_fold'][0] is None:
            datasource['val_fold'] *= len(datasource['test_fold'])

    if len(datasource['test_fold']) != len(datasource['val_fold']):
        print(Colors.red('ERROR: Need same number of validation and '
                         'test folds'))
        return 1

    all_pred_files = []
    all_gt_files = []

    print(Colors.magenta('\nStarting experiment ' + ex.observers[0].hash()))

    with TempDir() as exp_dir:
        for test_fold, val_fold in zip(datasource['test_fold'],
                                       datasource['val_fold']):
            print('')
            print(Colors.yellow(
                '=' * 20 + ' FOLD {} '.format(test_fold) + '=' * 20))
            # Load data sets
            print(Colors.red('\nLoading data...\n'))

            train_set, val_set, test_set, gt_files = data.create_datasources(
                dataset_names=datasource['datasets'],
                preprocessors=datasource['preprocessors'],
                compute_features=features.create_extractor(feature_extractor),
                compute_targets=target_computer,
                context_size=datasource['context_size'],
                test_fold=test_fold,
                val_fold=val_fold,
                cached=datasource['cached']
            )

            if testing['test_on_val']:
                test_set = val_set

            print(Colors.blue('Train Set:'))
            print('\t', train_set)

            print(Colors.blue('Validation Set:'))
            print('\t', val_set)

            print(Colors.blue('Test Set:'))
            print('\t', test_set)
            print('')

            # build network
            print(Colors.red('Building network...\n'))

            opt, learn_rate = create_optimiser(optimiser)

            neural_net = build_net(
                feature_shape=train_set.feature_shape,
                batch_size=None,
                optimiser=opt,
                out_size=train_set.target_shape[0],
                **net
            )

            print(Colors.blue('Neural Network:'))
            print(neural_net)
            print('')

            print(Colors.red('Starting training...\n'))

            updates = []
            if optimiser['schedule'] is not None:
                updates.append(
                    nn.LearnRateSchedule(
                        learn_rate=learn_rate, **optimiser['schedule'])
                )

            best_params, train_losses, val_losses = nn.train(
                neural_net, train_set, n_epochs=training['num_epochs'],
                batch_size=training['batch_size'], validation_set=val_set,
                early_stop=training['early_stop'],
                threaded=10,
                early_stop_acc=training['early_stop_acc'],
                updates=updates
            )

            print(Colors.red('\nStarting testing...\n'))

            param_file = os.path.join(
                exp_dir, 'params_fold_{}.pkl'.format(test_fold))
            neural_net.save_parameters(param_file)
            ex.add_artifact(param_file)

            pred_files = test.compute_labeling(
                neural_net, target_computer, test_set, dest_dir=exp_dir,
                rnn=False
            )

            test_gt_files = dmgr.files.match_files(
                pred_files, gt_files, test.PREDICTION_EXT, data.GT_EXT
            )

            all_pred_files += pred_files
            all_gt_files += test_gt_files

            print(Colors.blue('Results:'))
            scores = test.compute_average_scores(test_gt_files, pred_files)
            # convert to float so yaml output looks nice
            for k in scores:
                scores[k] = float(scores[k])
            test.print_scores(scores)

            result_file = os.path.join(
                exp_dir, 'results_fold_{}.yaml'.format(test_fold))
            yaml.dump(dict(scores=scores,
                           train_losses=map(float, train_losses),
                           val_losses=map(float, val_losses)),
                      open(result_file, 'w'))
            ex.add_artifact(result_file)

        # if there is something to aggregate
        if len(datasource['test_fold']) > 1:
            print(Colors.yellow('\nAggregated Results:\n'))
            scores = test.compute_average_scores(all_gt_files, all_pred_files)
            # convert to float so yaml output looks nice
            for k in scores:
                scores[k] = float(scores[k])
            test.print_scores(scores)
            result_file = os.path.join(exp_dir, 'results.yaml')
            yaml.dump(dict(scores=scores), open(result_file, 'w'))
            ex.add_artifact(result_file)

        for pf in all_pred_files:
            ex.add_artifact(pf)

    print(Colors.magenta('Stopping experiment ' + ex.observers[0].hash()))
