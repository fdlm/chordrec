from __future__ import print_function

import collections
import os

import theano.tensor as tt
import yaml
from sacred import Experiment

import data
import dmgr
import dnn
import features
import lasagne as lnn
import nn
import targets
import test
from exp_utils import PickleAndSymlinkObserver, TempDir, create_optimiser
from nn.utils import Colors

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


def stack_layers(net, batch_norm, conv1, conv2, conv3):

    for i, conv in enumerate([conv1, conv2, conv3]):
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


def stack_gap(net, out_size, num_filters, filter_size, dropout, batch_norm):
    net = lnn.layers.Conv2DLayer(
        net, num_filters=num_filters, filter_size=filter_size,
        pad=0, nonlinearity=lnn.nonlinearities.rectify,
        name='Gap_Filters')
    if batch_norm:
        net = lnn.layers.batch_norm(net)

    net = lnn.layers.DropoutLayer(net, p=dropout)

    net = lnn.layers.Conv2DLayer(
        net, num_filters=num_filters, filter_size=1,
        pad=0, nonlinearity=lnn.nonlinearities.rectify,
        name='Gap_Filters_Single')
    if batch_norm:
        net = lnn.layers.batch_norm(net)
    net = lnn.layers.DropoutLayer(net, p=dropout)

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


def build_net(feature_shape, out_size, net):

    if net['dense'] and net['global_avg_pool']:
        raise ValueError('Cannot use dense layers AND global avg. pool.')

    # input variables
    input_var = tt.tensor3('feature_input', dtype='float32')
    target_var = tt.matrix('target_output', dtype='float32')

    # stack more layers
    network = lnn.layers.InputLayer(
        name='input', shape=(None,) + feature_shape, input_var=input_var)

    # reshape to 1 "color" channel
    network = lnn.layers.reshape(
        network, shape=(-1, 1) + feature_shape, name='reshape')

    network = stack_layers(network, **net['conv'])

    if net['dense']:
        network = dnn.stack_layers(network, **net['dense'])
        # output classification layer
        network = lnn.layers.DenseLayer(
            network, name='output', num_units=out_size,
            nonlinearity=lnn.nonlinearities.softmax)
    elif net['global_avg_pool']:
        network = stack_gap(network, out_size, **net['global_avg_pool'])
    else:
        raise RuntimeError('Need to specify output architecture!')

    return network, input_var, target_var


@ex.config
def config():
    observations = 'results'

    datasource = dict(
            context_size=7,
    )

    feature_extractor = None

    target = None

    net = dict(
        conv=dict(
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
        ),
        global_avg_pool=None,
        dense=dict(
            num_layers=1,
            num_units=512,
            dropout=0.5,
            nonlinearity='rectify',
            batch_norm=False
        ),
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

    regularisation = dict(
        l2=1e-4,
        l1=0
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
            dropout=0.5,
            batch_norm=True
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
         regularisation, target, optimiser, training, testing):

    if feature_extractor is None:
        print(Colors.red('ERROR: Specify a feature extractor!'))
        return 1

    if target is None:
        print(Colors.red('ERROR: Specify a target!'))
        return 1

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

            neural_net, input_var, target_var = build_net(
                feature_shape=train_set.feature_shape,
                out_size=train_set.target_shape[0],
                net=net
            )

            opt, lrs = create_optimiser(optimiser)

            train_fn = nn.compile_train_fn(
                neural_net, input_var, target_var,
                loss_fn=compute_loss, opt_fn=opt,
                **regularisation
            )

            test_fn = nn.compile_test_func(
                neural_net, input_var, target_var,
                loss_fn=compute_loss,
                **regularisation
            )

            process_fn = nn.compile_process_func(neural_net, input_var)

            print(Colors.blue('Neural Network:'))
            print(nn.to_string(neural_net))
            print('')

            print(Colors.red('Starting training...\n'))

            train_losses, val_losses, val_accs = nn.train(
                network=neural_net,
                train_fn=train_fn, train_set=train_set,
                test_fn=test_fn, validation_set=val_set,
                threaded=10, updates=[lrs] if lrs else [],
                **training
            )

            print(Colors.red('\nStarting testing...\n'))

            param_file = os.path.join(
                exp_dir, 'params_fold_{}.pkl'.format(test_fold))
            nn.save_params(neural_net, param_file)
            ex.add_artifact(param_file)

            pred_files = test.compute_labeling(
                process_fn, target_computer, test_set, dest_dir=exp_dir,
                use_mask=False
            )

            test_gt_files = dmgr.files.match_files(
                pred_files, gt_files, test.PREDICTION_EXT, data.GT_EXT
            )

            all_pred_files += pred_files
            all_gt_files += test_gt_files

            print(Colors.blue('Results:'))
            scores = test.compute_average_scores(test_gt_files, pred_files)
            test.print_scores(scores)
            result_file = os.path.join(
                exp_dir, 'results_fold_{}.yaml'.format(test_fold))
            yaml.dump(dict(scores=scores,
                           train_losses=map(float, train_losses),
                           val_losses=map(float, val_losses),
                           val_accs=map(float, val_accs)),
                      open(result_file, 'w'))
            ex.add_artifact(result_file)

        # if there is something to aggregate
        if len(datasource['test_fold']) > 1:
            print(Colors.yellow('\nAggregated Results:\n'))
            scores = test.compute_average_scores(all_gt_files, all_pred_files)
            test.print_scores(scores)
            result_file = os.path.join(exp_dir, 'results.yaml')
            yaml.dump(dict(scores=scores), open(result_file, 'w'))
            ex.add_artifact(result_file)

        for pf in all_pred_files:
            ex.add_artifact(pf)

    print(Colors.magenta('Stopping experiment ' + ex.observers[0].hash()))
