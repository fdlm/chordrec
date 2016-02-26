from __future__ import print_function
import os
import theano
import theano.tensor as tt
import lasagne as lnn
import yaml
import collections
from sacred import Experiment


import nn
import dmgr
from nn.utils import Colors

import test
import data
import features
import targets
from exp_utils import (PickleAndSymlinkObserver, TempDir, create_optimiser,
                       ParamSaver)

# Initialise Sacred experiment
ex = Experiment('Deep Neural Network')
ex.observers.append(PickleAndSymlinkObserver())
data.add_sacred_config(ex)
features.add_sacred_config(ex)
targets.add_sacred_config(ex)


def compute_loss(prediction, target):
    # need to clip predictions for numerical stability
    eps = 1e-7
    pred_clip = tt.clip(prediction, eps, 1.-eps)
    return lnn.objectives.categorical_crossentropy(pred_clip, target).mean()


def stack_layers(inp, nonlinearity, num_layers, num_units, dropout,
                 batch_norm):

    nl = getattr(lnn.nonlinearities, nonlinearity)
    net = inp

    for i in range(num_layers):
        if batch_norm:
            net = lnn.layers.batch_norm(
                lnn.layers.DenseLayer(
                    net, num_units=num_units, nonlinearity=nl,
                    name='fc-{}'.format(i)
                )
            )
        else:
            net = lnn.layers.DenseLayer(
                net, num_units=num_units, nonlinearity=nl,
                name='fc-{}'.format(i)
            )
        if dropout > 0.0:
            net = lnn.layers.DropoutLayer(net, p=dropout)

    return net


def build_net(feature_shape, optimiser, out_size, l2, num_units, num_layers,
              dropout, batch_norm, nonlinearity):
    # input variables
    feature_var = (tt.tensor3('feature_input', dtype='float32')
                   if len(feature_shape) > 1 else
                   tt.matrix('feature_input', dtype='float32'))
    target_var = tt.matrix('target_output', dtype='float32')

    # stack more layers
    net = lnn.layers.InputLayer(name='input',
                                shape=(None,) + feature_shape,
                                input_var=feature_var)

    net = stack_layers(net, nonlinearity, num_layers, num_units, dropout,
                       batch_norm)

    # output layer
    net = lnn.layers.DenseLayer(net, name='output', num_units=out_size,
                                nonlinearity=lnn.nonlinearities.softmax)

    # create train function
    prediction = lnn.layers.get_output(net)
    l2_penalty = lnn.regularization.regularize_network_params(
        net, lnn.regularization.l2) * l2
    loss = compute_loss(prediction, target_var) + l2_penalty
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
        num_layers=3,
        num_units=256,
        dropout=0.5,
        nonlinearity='rectify',
        batch_norm=False,
        l2=1e-4,
    )

    optimiser = dict(
        name='adam',
        params=dict(
            learning_rate=0.0001
        )
    )

    training = dict(
        num_epochs=500,
        early_stop=20,
        batch_size=512,
        early_stop_acc=True,
    )


@ex.named_config
def no_context():
    datasource = dict(
        context_size=0
    )

    net = dict(
        num_units=100,
        dropout=0.3,
        l2_lambda=0.
    )


@ex.automain
def main(_config, _run, observations, datasource, net, feature_extractor,
         target, optimiser, training):

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

    all_pred_files = []
    all_gt_files = []

    with TempDir() as exp_dir:
        for test_fold in datasource['test_fold']:
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
                val_fold=datasource['val_fold'],
                cached=datasource['cached']
            )

            print(Colors.blue('Train Set:'))
            print('\t', train_set)
            print(Colors.blue('Validation Set:'))
            print('\t', val_set)
            print(Colors.blue('Test Set:'))
            print('\t', test_set)
            print('')

            # build network
            print(Colors.red('Building network...\n'))

            neural_net = build_net(
                feature_shape=train_set.feature_shape,
                optimiser=create_optimiser(optimiser),
                out_size=train_set.target_shape[0],
                **net
            )

            print(Colors.blue('Neural Network:'))
            print(neural_net)
            print('')

            print(Colors.red('Starting training...\n'))

            best_params, train_losses, val_losses = nn.train(
                neural_net, train_set, n_epochs=training['num_epochs'],
                batch_size=training['batch_size'], validation_set=val_set,
                early_stop=training['early_stop'],
                early_stop_acc=training['early_stop_acc'],
                threaded=10,
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

        for pf in all_pred_files:
            ex.add_artifact(pf)

    print('')
