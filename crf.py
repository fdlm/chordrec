from __future__ import print_function
import os
import theano
import theano.tensor as tt
import lasagne as lnn
import numpy as np
import spaghetti as spg
import yaml
from sacred import Experiment

import nn
import dmgr
from nn.utils import Colors

import test
import data
import features
from plotting import CrfPlotter
from exp_utils import PickleAndSymlinkObserver, TempDir, create_optimiser

# Initialise Sacred experiment
ex = Experiment('Conditional Random Field')
ex.observers.append(PickleAndSymlinkObserver())
data.add_sacred_config(ex)
features.add_sacred_config(ex)


def compute_loss(network, target, mask):
    loss = spg.objectives.neg_log_likelihood(network, target, mask)
    return lnn.objectives.aggregate(loss, mode='mean')


def add_dense_layers(net, feature_shape, true_batch_size, true_seq_len,
                     num_units, num_layers, dropout):

    net = lnn.layers.ReshapeLayer(net, (-1,) + feature_shape,
                                  name='reshape to single')

    for i in range(num_layers):
        net = lnn.layers.DenseLayer(net, num_units=num_units,
                                    name='fc-{}'.format(i))
        net = lnn.layers.DropoutLayer(net, p=dropout)

    net = lnn.layers.ReshapeLayer(
        net, (true_batch_size, true_seq_len, num_units),
        name='reshape to sequence'
    )

    return net


def add_recurrent_layers(net, mask_in, num_units, num_layers, grad_clip,
                         dropout, bidirectional):
    fwd = net
    for i in range(num_layers):
        fwd = lnn.layers.RecurrentLayer(
            fwd, name='recurrent_fwd_{}'.format(i),
            num_units=num_units, mask_input=mask_in,
            grad_clipping=grad_clip,
            W_in_to_hid=lnn.init.GlorotUniform(),
            W_hid_to_hid=lnn.init.Orthogonal(gain=np.sqrt(2) / 2),
        )
        fwd = lnn.layers.DropoutLayer(fwd, p=dropout)

    if not bidirectional:
        return fwd
    else:
        bck = net
        for i in range(num_layers):
            bck = lnn.layers.RecurrentLayer(
                    bck, name='recurrent_bck_{}'.format(i),
                    num_units=num_units, mask_input=mask_in,
                    grad_clipping=grad_clip,
                    W_in_to_hid=lnn.init.GlorotUniform(),
                    W_hid_to_hid=lnn.init.Orthogonal(gain=np.sqrt(2) / 2),
                    backwards=True
            )
            bck = lnn.layers.DropoutLayer(bck, p=dropout)

        # combine the forward and backward recurrent layers...
        return lnn.layers.ConcatLayer([fwd, bck], name='fwd + bck', axis=-1)


def build_net(feature_shape, batch_size, l2_lambda, max_seq_len,
              dense, recurrent, optimiser, out_size):

    # input variables
    feature_var = (tt.tensor4('feature_input', dtype='float32')
                   if len(feature_shape) > 1 else
                   tt.tensor3('feature_input', dtype='float32'))

    target_var = tt.tensor3('target_output', dtype='float32')
    mask_var = tt.matrix('mask_input', dtype='float32')

    # stack more layers (in this case create a CRF)
    net = lnn.layers.InputLayer(
        name='input', shape=(batch_size, max_seq_len) + feature_shape,
        input_var=feature_var
    )

    mask_in = lnn.layers.InputLayer(
        name='mask', input_var=mask_var, shape=(batch_size, max_seq_len)
    )

    # add dense layers between input and crf
    if dense['num_layers'] > 0:
        true_batch_size, true_seq_len = feature_var.shape[:2]
        net = add_dense_layers(
            net, feature_shape, true_batch_size, true_seq_len, **dense
        )

    # add recurrent layers between input and crf
    if recurrent['num_layers'] > 0:
        net = add_recurrent_layers(net, mask_in, **recurrent)

    net = spg.layers.CrfLayer(incoming=net, mask_input=mask_in,
                              num_states=out_size, name='CRF')

    # create train function - this one uses the log-likelihood objective
    l2_penalty = lnn.regularization.regularize_network_params(
        net, lnn.regularization.l2) * l2_lambda
    loss = compute_loss(net, target_var, mask_var) + l2_penalty
    params = lnn.layers.get_all_params(net, trainable=True)
    updates = optimiser(loss, params)
    train = theano.function([feature_var, mask_var, target_var], loss,
                            updates=updates)

    # create test and process function. process just computes the prediction
    # without computing the loss, and thus does not need target labels
    test_loss = compute_loss(net, target_var, mask_var) + l2_penalty
    test = theano.function([feature_var, mask_var, target_var], test_loss)
    viterbi_out = lnn.layers.get_output(net, mode='viterbi')
    process = theano.function([feature_var, mask_var], viterbi_out)

    return nn.NeuralNetwork(net, train, test, process)


@ex.config
def config():
    observations = 'results'

    plot = False

    datasource = dict(
        context_size=3
    )

    feature_extractor = None

    net = dict(
        l2_lambda=1e-4,
        dense=dict(num_layers=0),
        recurrent=dict(num_layers=0)
    )

    optimiser = dict(
        name='adam',
        params=dict(
            learning_rate=0.002
        )
    )

    training = dict(
        num_epochs=1000,
        early_stop=20,
        batch_size=64,
        max_seq_len=1024  # at 10 fps, this corresponds to 102 seconds
    )


@ex.named_config
def dense_input():
    net = dict(
        dense=dict(
            num_layers=3,
            num_units=256,
            dropout=0.5
        )
    )


@ex.named_config
def recurrent_input():
    net = dict(
        recurrent=dict(
            num_units=128,
            num_layers=3,
            dropout=0.3,
            grad_clip=1.,
            bidirectional=True
        )
    )


@ex.automain
def main(_config, _run, observations, datasource, net, feature_extractor,
         optimiser, training, plot):

    if feature_extractor is None:
        print(Colors.red('ERROR: Specify a feature extractor!'))
        return 1

    # Load data sets
    print(Colors.red('Loading data...\n'))

    train_set, val_set, test_set, gt_files = data.create_datasources(
        dataset_names=datasource['datasets'],
        preprocessors=datasource['preprocessors'],
        compute_features=features.create_extractor(feature_extractor),
        compute_targets=data.ChordsMajMin(feature_extractor['params']['fps']),
        context_size=datasource['context_size'],
        test_fold=datasource['test_fold'],
        val_fold=datasource['val_fold']
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

    train_neural_net = build_net(
        feature_shape=train_set.feature_shape,
        batch_size=training['batch_size'],
        max_seq_len=training['max_seq_len'],
        l2_lambda=net['l2_lambda'],
        dense=net['dense'],
        recurrent=net['recurrent'],
        optimiser=create_optimiser(optimiser),
        out_size=train_set.target_shape[0]
    )

    print(Colors.blue('Neural Network:'))
    print(train_neural_net)
    print('')

    print(Colors.red('Starting training...\n'))

    with TempDir() as dest_dir:

        if plot:
            plot_file = os.path.join(dest_dir, 'plot.pdf')
            updates = [CrfPlotter(train_neural_net.network, plot_file)]
        else:
            updates = []

        best_params, train_losses, val_losses = nn.train(
            train_neural_net, train_set, n_epochs=training['num_epochs'],
            batch_size=training['batch_size'], validation_set=val_set,
            early_stop=training['early_stop'],
            batch_iterator=dmgr.iterators.iterate_datasources,
            sequence_length=training['max_seq_len'],
            updates=updates,
            threaded=10
        )

        if plot:
            updates[-1].close()
            ex.add_artifact(plot_file)

        print(Colors.red('\nStarting testing...\n'))

        del train_neural_net

        # build test crf with batch size 1 and no max sequence length
        test_neural_net = build_net(
            feature_shape=test_set.feature_shape,
            batch_size=1,
            max_seq_len=None,
            l2_lambda=net['l2_lambda'],
            dense=net['dense'],
            recurrent=net['recurrent'],
            optimiser=create_optimiser(optimiser),
            out_size=test_set.target_shape[0]
        )

        # load previously learnt parameters
        test_neural_net.set_parameters(best_params)

        param_file = os.path.join(dest_dir, 'params.pkl')
        test_neural_net.save_parameters(param_file)
        ex.add_artifact(param_file)

        pred_files = test.compute_labeling(
            test_neural_net, test_set, dest_dir=dest_dir,
            fps=feature_extractor['params']['fps'], rnn=True, out_onehot=False
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

