from __future__ import print_function
import os
import collections
import numpy as np
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
import features
import targets
from exp_utils import PickleAndSymlinkObserver, TempDir, create_optimiser

# Initialise Sacred experiment
ex = Experiment('Recurrent Neural Network')
ex.observers.append(PickleAndSymlinkObserver())
data.add_sacred_config(ex)
features.add_sacred_config(ex)
targets.add_sacred_config(ex)


def compute_loss(prediction, target, mask):
    # need to clip predictions for numerical stability
    eps = 1e-7
    pred_clip = tt.clip(prediction, eps, 1.-eps)
    loss = lnn.objectives.categorical_crossentropy(pred_clip, target)
    return lnn.objectives.aggregate(loss, mask, mode='normalized_sum')


def build_net(feature_shape, out_size, optimiser, l2, num_rec_units,
              num_layers, dropout, grad_clip, nonlinearity, bidirectional):
    # input variables
    feature_var = tt.tensor3('feature_input', dtype='float32')
    target_var = tt.tensor3('target_output', dtype='float32')
    mask_var = tt.matrix('mask_input', dtype='float32')

    # stack more layers
    net = lnn.layers.InputLayer(
            name='input', shape=(None, None) + feature_shape,
            input_var=feature_var
    )

    true_batch_size, true_seq_len, _ = feature_var.shape

    mask_in = lnn.layers.InputLayer(name='mask',
                                    input_var=mask_var,
                                    shape=(None, None))

    if nonlinearity != 'LSTM':
        nl = getattr(lnn.nonlinearities, nonlinearity)

        def add_layer(prev_layer, **kwargs):
            return lnn.layers.RecurrentLayer(
                prev_layer, num_units=num_rec_units, mask_input=mask_in,
                grad_clipping=grad_clip, nonlinearity=nl,
                W_in_to_hid=lnn.init.GlorotUniform(),
                W_hid_to_hid=lnn.init.Orthogonal(gain=np.sqrt(2) / 2),
                **kwargs)

    else:
        def add_layer(prev_layer, **kwargs):
            return lnn.layers.LSTMLayer(
                prev_layer, num_units=num_rec_units, mask_input=mask_in,
                grad_clipping=grad_clip,
                **kwargs
            )

    fwd = net
    for i in range(num_layers):
        fwd = add_layer(fwd, name='rec_fwd_{}'.format(i))
        if dropout > 0.:
            fwd = lnn.layers.DropoutLayer(fwd, p=dropout)

    if not bidirectional:
        net = fwd
    else:
        bck = net
        for i in range(num_layers):
            bck = add_layer(bck, name='rec_bck_{}'.format(i), backwards=True)
            if dropout > 0:
                bck = lnn.layers.DropoutLayer(bck, p=dropout)

        # combine the forward and backward recurrent layers...
        net = lnn.layers.ConcatLayer([fwd, bck], name='fwd + bck', axis=-1)

    # In order to connect a recurrent layer to a dense layer, we need to
    # flatten the first two dimensions (our "sample dimensions"); this will
    # cause each time step of each sequence to be processed independently
    net = lnn.layers.ReshapeLayer(net, (-1, num_rec_units * 2),
                                  name='reshape to single')

    net = lnn.layers.DenseLayer(net, num_units=out_size,
                                nonlinearity=lnn.nonlinearities.softmax,
                                name='output')
    # To reshape back to our original shape, we can use the symbolic shape
    # variables we retrieved above.
    net = lnn.layers.ReshapeLayer(net,
                                  (true_batch_size, true_seq_len, out_size),
                                  name='output-reshape')

    # create train function
    prediction = lnn.layers.get_output(net)
    l2_penalty = lnn.regularization.regularize_network_params(
        net, lnn.regularization.l2) * l2
    loss = compute_loss(prediction, target_var, mask_var) + l2_penalty
    params = lnn.layers.get_all_params(net, trainable=True)
    updates = optimiser(loss, params)
    train = theano.function([feature_var, mask_var, target_var], loss,
                            updates=updates)

    # create test and process function. process just computes the prediction
    # without computing the loss, and thus does not need target labels
    test_prediction = lnn.layers.get_output(net, deterministic=True)
    test_loss = (compute_loss(test_prediction, target_var, mask_var) +
                 l2_penalty)
    test = theano.function([feature_var, mask_var, target_var],
                           [test_loss, test_prediction])
    process = theano.function([feature_var, mask_var], test_prediction)

    return nn.NeuralNetwork(net, train, test, process)


@ex.config
def config():
    observations = 'results'

    feature_extractor = None

    target = None

    net = dict(
        l2=1e-4,
        num_rec_units=128,
        num_layers=3,
        dropout=0.3,
        grad_clip=1.,
        bidirectional=True,
        nonlinearity='rectify'
    )

    optimiser = dict(
        name='adam',
        params=dict(
            learning_rate=0.001
        ),
        schedule=None
    )

    training = dict(
        num_epochs=1000,
        early_stop=20,
        early_stop_acc=True,
        batch_size=64,
        max_seq_len=1024
    )

    testing = dict(
        test_on_val=False
    )


@ex.named_config
def lstm():
    net = dict(
        nonlinearity='LSTM',
        num_rec_units=64,
    )


@ex.automain
def main(_config, _run, observations, datasource, net, feature_extractor,
         target, optimiser, training, testing):

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
                context_size=datasource['context_size'],
                compute_targets=target_computer,
                test_fold=test_fold,
                val_fold=val_fold,
                cached=datasource['cached'],
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
                out_size=train_set.target_shape[0],
                optimiser=opt,
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
                batch_iterator=dmgr.iterators.iterate_datasources,
                sequence_length=training['max_seq_len'],
                threaded=10,
                updates=updates
            )

            print(Colors.red('\nStarting testing...\n'))

            param_file = os.path.join(
                exp_dir, 'params_fold_{}.pkl'.format(test_fold))
            neural_net.save_parameters(param_file)
            ex.add_artifact(param_file)

            pred_files = test.compute_labeling(
                neural_net, target_computer, test_set, dest_dir=exp_dir,
                rnn=True
            )

            test_gt_files = dmgr.files.match_files(
                pred_files, gt_files, test.PREDICTION_EXT, data.GT_EXT
            )

            all_pred_files += pred_files
            all_gt_files += test_gt_files

            print(Colors.red('\nResults:\n'))
            scores = test.compute_average_scores(test_gt_files, pred_files)
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
            test.print_scores(scores)
            result_file = os.path.join(exp_dir, 'results.yaml')
            yaml.dump(dict(scores=scores), open(result_file, 'w'))
            ex.add_artifact(result_file)

        for pf in all_pred_files:
            ex.add_artifact(pf)

    print(Colors.magenta('Stopping experiment ' + ex.observers[0].hash()))
