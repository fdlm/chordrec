from __future__ import print_function

import collections
import os

import numpy as np
import theano
import theano.tensor as tt
import yaml
from sacred import Experiment

import data
import dmgr
import features
import lasagne as lnn
import nn
import targets
import test
from exp_utils import PickleAndSymlinkObserver, TempDir, create_optimiser
from nn.utils import Colors

# Initialise Sacred experiment
ex = Experiment('Recurrent Neural Network')
ex.observers.append(PickleAndSymlinkObserver())
data.add_sacred_config(ex)
features.add_sacred_config(ex)
targets.add_sacred_config(ex)


def compute_loss(prediction, target, mask):
    # need to clip predictions for numerical stability
    eps = 1e-7
    pred_clip = tt.clip(prediction, eps, 1. - eps)
    loss = lnn.objectives.categorical_crossentropy(pred_clip, target)
    return lnn.objectives.aggregate(loss, mask, mode='normalized_sum')


def stack_layers(net, mask_in, num_rec_units, num_layers, dropout, grad_clip,
                 bidirectional, nonlinearity):

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
        return net

    bck = net
    for i in range(num_layers):
        bck = add_layer(bck, name='rec_bck_{}'.format(i), backwards=True)
        if dropout > 0:
            bck = lnn.layers.DropoutLayer(bck, p=dropout)

    # combine the forward and backward recurrent layers...
    net = lnn.layers.ConcatLayer([fwd, bck], name='fwd + bck', axis=-1)
    return net


def build_net(feature_shape, out_size, net):
    # input variables
    input_var = tt.tensor3('input', dtype='float32')
    target_var = tt.tensor3('target_output', dtype='float32')
    mask_var = tt.matrix('mask_input', dtype='float32')

    # stack more layers
    network = lnn.layers.InputLayer(
        name='input', shape=(None, None) + feature_shape,
        input_var=input_var
    )

    true_batch_size, true_seq_len, _ = input_var.shape

    mask_in = lnn.layers.InputLayer(name='mask',
                                    input_var=mask_var,
                                    shape=(None, None))

    network = stack_layers(network, mask_in, **net)

    # In order to connect a recurrent layer to a dense layer, we need to
    # flatten the first two dimensions (our "sample dimensions"); this will
    # cause each time step of each sequence to be processed independently
    network = lnn.layers.ReshapeLayer(
        network, (-1, lnn.layers.get_output_shape(network)[-1]),
        name='reshape to single')

    network = lnn.layers.DenseLayer(
        network, num_units=out_size, nonlinearity=lnn.nonlinearities.softmax,
        name='output')

    # To reshape back to our original shape, we can use the symbolic shape
    # variables we retrieved above.
    network = lnn.layers.ReshapeLayer(
        network, (true_batch_size, true_seq_len, out_size),
        name='output-reshape')

    return network, input_var, mask_var, target_var


@ex.config
def config():
    observations = 'results'

    feature_extractor = None

    target = None

    net = dict(
        num_rec_units=128,
        num_layers=3,
        dropout=0.3,
        grad_clip=0,
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
        batch_size=32,
        max_seq_len=1024
    )

    regularisation = dict(
        l1=0.0,
        l2=1e-8,
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

            neural_net, input_var, mask_var, target_var = build_net(
                feature_shape=train_set.feature_shape,
                out_size=train_set.target_shape[0],
                net=net
            )

            opt, lrs = create_optimiser(optimiser)

            train_fn = nn.compile_train_fn(
                neural_net, input_var, target_var,
                loss_fn=compute_loss, opt_fn=opt, mask_var=mask_var,
                **regularisation
            )

            test_fn = nn.compile_test_func(
                neural_net, input_var, target_var,
                loss_fn=compute_loss, mask_var=mask_var,
                **regularisation
            )

            process_fn = nn.compile_process_func(neural_net, input_var,
                                                 mask_var)
            print(Colors.blue('Neural Network:'))
            print(nn.to_string(neural_net))
            print('')

            print(Colors.red('Starting training...\n'))

            train_losses, val_losses, val_accs = nn.train(
                network=neural_net,
                train_fn=train_fn, train_set=train_set,
                test_fn=test_fn, validation_set=val_set,
                threaded=10, updates=[lrs] if lrs else [],
                batch_iterator=dmgr.iterators.iterate_datasources,
                **training
            )

            print(Colors.red('\nStarting testing...\n'))

            param_file = os.path.join(
                exp_dir, 'params_fold_{}.pkl'.format(test_fold))
            nn.save_params(neural_net, param_file)
            ex.add_artifact(param_file)

            pred_files = test.compute_labeling(
                process_fn, target_computer, test_set, dest_dir=exp_dir,
                use_mask=True
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
