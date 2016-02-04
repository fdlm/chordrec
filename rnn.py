from __future__ import print_function
import os
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
from exp_utils import PickleAndSymlinkObserver, TempDir, create_optimiser

# Initialise Sacred experiment
ex = Experiment('Recurrent Neural Network')
ex.observers.append(PickleAndSymlinkObserver())
data.add_sacred_config(ex)
features.add_sacred_config(ex)


def compute_loss(prediction, target, mask):
    # need to clip predictions for numerical stability
    eps = 1e-7
    pred_clip = tt.clip(prediction, eps, 1.-eps)
    loss = lnn.objectives.categorical_crossentropy(pred_clip, target)
    return lnn.objectives.aggregate(loss, mask, mode='normalized_sum')


def build_net(feature_shape, batch_size, l2_lambda, num_rec_units,
              num_layers, dropout, grad_clip, bidirectional, optimiser,
              max_seq_len, out_size):
    # input variables
    feature_var = tt.tensor3('feature_input', dtype='float32')
    target_var = tt.tensor3('target_output', dtype='float32')
    mask_var = tt.matrix('mask_input', dtype='float32')

    # stack more layers
    # We do not know the length of the sequence...
    net = lnn.layers.InputLayer(
            name='input', shape=(batch_size, max_seq_len) + feature_shape,
            input_var=feature_var
    )

    true_batch_size, true_seq_len, _ = feature_var.shape

    mask_in = lnn.layers.InputLayer(name='mask',
                                    input_var=mask_var,
                                    shape=(batch_size, max_seq_len))

    fwd = net
    for i in range(num_layers):
        fwd = lnn.layers.RecurrentLayer(
            fwd, name='recurrent_fwd_{}'.format(i),
            num_units=num_rec_units, mask_input=mask_in,
            grad_clipping=grad_clip,
            W_in_to_hid=lnn.init.GlorotUniform(),
            W_hid_to_hid=lnn.init.Orthogonal(gain=np.sqrt(2) / 2),
        )
        fwd = lnn.layers.DropoutLayer(fwd, p=dropout)

    if not bidirectional:
        net = fwd
    else:
        bck = net
        for i in range(num_layers):
            bck = lnn.layers.RecurrentLayer(
                bck, name='recurrent_bck_{}'.format(i),
                num_units=num_rec_units, mask_input=mask_in,
                grad_clipping=grad_clip,
                W_in_to_hid=lnn.init.GlorotUniform(),
                W_hid_to_hid=lnn.init.Orthogonal(gain=np.sqrt(2) / 2),
                backwards=True
            )
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
        net, lnn.regularization.l2) * l2_lambda
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
    test = theano.function([feature_var, mask_var, target_var], test_loss)
    process = theano.function([feature_var, mask_var], test_prediction)

    return nn.NeuralNetwork(net, train, test, process)


@ex.config
def config():
    observations = 'results'

    feature_extractor = None

    net = dict(
        l2_lambda=1e-4,
        num_rec_units=128,
        num_layers=3,
        dropout=0.3,
        grad_clip=1.,
        bidirectional=True
    )

    optimiser = dict(
        name='adam',
        params=dict(
            learning_rate=0.001
        )
    )

    training = dict(
        num_epochs=1000,
        early_stop=20,
        batch_size=64,
        max_seq_len=1024
    )


@ex.automain
def main(_config, _run, observations, datasource, net, feature_extractor,
         optimiser, training):

    if feature_extractor is None:
        print(Colors.red('ERROR: Specify a feature extractor!'))
        return 1

    # Load data sets
    print(Colors.red('Loading data...\n'))

    train_set, val_set, test_set, gt_files = data.create_datasources(
        dataset_names=datasource['datasets'],
        preprocessors=datasource['preprocessors'],
        compute_features=features.create_extractor(feature_extractor),
        compute_targets=data.chords_maj_min,
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
        num_rec_units=net['num_rec_units'],
        num_layers=net['num_layers'],
        dropout=net['dropout'],
        bidirectional=net['bidirectional'],
        grad_clip=net['grad_clip'],
        optimiser=create_optimiser(optimiser),
        out_size=train_set.target_shape[0]
    )

    print(Colors.blue('Neural Network:'))
    print(train_neural_net)
    print('')

    print(Colors.red('Starting training...\n'))

    best_params, train_losses, val_losses = nn.train(
        train_neural_net, train_set, n_epochs=training['num_epochs'],
        batch_size=training['batch_size'], validation_set=val_set,
        early_stop=training['early_stop'],
        batch_iterator=dmgr.iterators.iterate_datasources,
        sequence_length=training['max_seq_len'],
        threaded=10
    )

    print(Colors.red('\nStarting testing...\n'))

    del train_neural_net

    # build test rnn with batch size 1 and no max sequence length
    test_neural_net = build_net(
        feature_shape=test_set.feature_shape,
        batch_size=1,
        max_seq_len=None,
        l2_lambda=net['l2_lambda'],
        num_rec_units=net['num_rec_units'],
        num_layers=net['num_layers'],
        dropout=net['dropout'],
        bidirectional=net['bidirectional'],
        grad_clip=net['grad_clip'],
        optimiser=create_optimiser(optimiser),
        out_size=test_set.target_shape[0]
    )

    # load previously learnt parameters
    test_neural_net.set_parameters(best_params)

    with TempDir() as dest_dir:
        param_file = os.path.join(dest_dir, 'params.pkl')
        test_neural_net.save_parameters(param_file)
        ex.add_artifact(param_file)

        pred_files = test.compute_labeling(
            test_neural_net, test_set, dest_dir=dest_dir,
            fps=feature_extractor['params']['fps'], rnn=True
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

