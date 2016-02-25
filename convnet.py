from __future__ import print_function
import os
import theano
import theano.tensor as tt
import lasagne as lnn
import yaml
from sacred import Experiment


import nn
import dmgr
from nn.utils import Colors

import test
import data
import targets
import features
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


def build_net(feature_shape, batch_size, l2_lambda, conv1, pool1, conv2,
              pool2, dense, optimiser, out_size):

    # input variables
    feature_var = tt.tensor3('feature_input', dtype='float32')
    target_var = tt.matrix('target_output', dtype='float32')

    # stack more layers
    net = lnn.layers.InputLayer(name='input',
                                shape=(batch_size,) + feature_shape,
                                input_var=feature_var)

    net = lnn.layers.reshape(net, shape=(-1, 1) + feature_shape,
                             name='reshape')

    for i in range(conv1['num_layers']):
        net = lnn.layers.Conv2DLayer(
            net, num_filters=conv1['num_filters'],
            filter_size=conv1['filter_size'],  # pad='same', for 1st layer
            nonlinearity=lnn.nonlinearities.rectify,
            name='Conv_1_{}'.format(i))

    net = lnn.layers.MaxPool2DLayer(net, pool_size=pool1['size'],
                                    name='Pool_1')
    if pool1['dropout'] > 0.0:
        net = lnn.layers.DropoutLayer(net, p=pool1['dropout'])

    for i in range(conv2['num_layers']):
        net = lnn.layers.Conv2DLayer(
                net, num_filters=conv2['num_filters'],
                filter_size=conv2['filter_size'],  # pad='same', for 1st layer
                nonlinearity=lnn.nonlinearities.rectify,
                name='Conv_2_{}'.format(i))

    net = lnn.layers.MaxPool2DLayer(net, pool_size=pool2['size'],
                                    name='Pool_2')
    if pool2['dropout'] > 0.0:
        net = lnn.layers.DropoutLayer(net, p=pool2['dropout'])

    for i in range(dense['num_layers']):
        net = lnn.layers.DenseLayer(net, num_units=dense['num_units'],
                                    nonlinearity=lnn.nonlinearities.rectify,
                                    name='Dense_{}'.format(i))
        if dense['dropout'] > 0.0:
            net = lnn.layers.DropoutLayer(net, p=dense['dropout'])

    # output layer
    net = lnn.layers.DenseLayer(net, name='output', num_units=out_size,
                                nonlinearity=lnn.nonlinearities.softmax)

    # create train function
    prediction = lnn.layers.get_output(net)
    l2_penalty = lnn.regularization.regularize_network_params(
            net, lnn.regularization.l2) * l2_lambda
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
        conv1=dict(
            num_layers=2,
            num_filters=32,
            filter_size=(3, 3),
        ),
        pool1=dict(
            size=(1, 2),
            dropout=0.25
        ),
        conv2=dict(
            num_layers=1,
            num_filters=64,
            filter_size=(3, 3)
        ),
        pool2=dict(
            size=(1, 2),
            dropout=0.25
        ),
        dense=dict(
            num_layers=1,
            num_units=512,
            dropout=0.5
        ),
        l2_lambda=1e-4
    )

    optimiser = dict(
        name='adam',
        params=dict(
                learning_rate=0.001
        )
    )

    training = dict(
        num_epochs=500,
        early_stop=20,
        early_stop_acc=True,
        batch_size=512,
    )


@ex.automain
def main(_config, _run, observations, datasource, net, feature_extractor,
         target, optimiser, training):

    if feature_extractor is None:
        print(Colors.red('ERROR: Specify a feature extractor!'))
        return 1

    # Load data sets
    print(Colors.red('Loading data...\n'))

    target_computer = targets.create_target(
        feature_extractor['params']['fps'],
        target
    )
    train_set, val_set, test_set, gt_files = data.create_datasources(
        dataset_names=datasource['datasets'],
        preprocessors=datasource['preprocessors'],
        compute_features=features.create_extractor(feature_extractor),
        compute_targets=target_computer,
        context_size=datasource['context_size'],
        test_fold=datasource['test_fold'],
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
        batch_size=None,
        l2_lambda=net['l2_lambda'],
        conv1=net['conv1'],
        conv2=net['conv2'],
        pool1=net['pool1'],
        pool2=net['pool2'],
        dense=net['dense'],
        optimiser=create_optimiser(optimiser),
        out_size=train_set.target_shape[0]
    )

    print(Colors.blue('Neural Network:'))
    print(neural_net)
    print('')

    print(Colors.red('Starting training...\n'))

    best_params, train_losses, val_losses = nn.train(
        neural_net, train_set, n_epochs=training['num_epochs'],
        batch_size=training['batch_size'], validation_set=val_set,
        early_stop=training['early_stop'],
        threaded=10,
        early_stop_acc=training['early_stop_acc']
    )

    print(Colors.red('\nStarting testing...\n'))

    neural_net.set_parameters(best_params)

    with TempDir() as dest_dir:
        param_file = os.path.join(dest_dir, 'params.pkl')
        neural_net.save_parameters(param_file)
        ex.add_artifact(param_file)

        pred_files = test.compute_labeling(
            neural_net, target_computer, test_set, dest_dir=dest_dir,
            rnn=False
        )

        test_gt_files = dmgr.files.match_files(
            pred_files, gt_files, test.PREDICTION_EXT, data.GT_EXT
        )

        print(Colors.red('\nResults:\n'))
        scores = test.compute_average_scores(test_gt_files, pred_files)
        # convert to float so yaml output looks nice
        for k in scores:
            scores[k] = float(scores[k])
        test.print_scores(scores)

        result_file = os.path.join(dest_dir, 'results.yaml')
        yaml.dump(dict(scores=scores,
                       train_losses=map(float, train_losses),
                       val_losses=map(float, val_losses)),
                  open(result_file, 'w'))
        ex.add_artifact(result_file)

        for pf in pred_files:
            ex.add_artifact(pf)

    print('')
