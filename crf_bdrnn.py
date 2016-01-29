from __future__ import print_function
import os
import numpy as np
import theano
import theano.tensor as tt
import lasagne as lnn
import spaghetti as spg

import nn
import dmgr

import data
import test

from nn.utils import Colors


def stack_layers(feature_var, mask_var, feature_shape, batch_size, max_seq_len,
                 out_size):
    # We do not know the length of the sequence...
    net = lnn.layers.InputLayer(name='input',
                                shape=(batch_size, max_seq_len) + feature_shape,
                                input_var=feature_var)

    true_batch_size, true_seq_len, _ = feature_var.shape

    mask_in = lnn.layers.InputLayer(name='mask',
                                    input_var=mask_var,
                                    shape=(batch_size, max_seq_len))

    n_rec_units = 128

    fwd = lnn.layers.RecurrentLayer(
            net, name='recurrent_fwd_1', num_units=n_rec_units, mask_input=mask_in,
            grad_clipping=1.,
            W_in_to_hid=lnn.init.HeUniform(),
            W_hid_to_hid=lnn.init.HeUniform(),
            # W_hid_to_hid=np.eye(n_rec_units, dtype=np.float32) * 0.9
    )

    bck = lnn.layers.RecurrentLayer(
            net, name='recurrent_bck_1', num_units=n_rec_units, mask_input=mask_in,
            grad_clipping=1.,
            W_in_to_hid=lnn.init.HeUniform(),
            W_hid_to_hid=lnn.init.HeUniform(),
            # learn_init=True,
            # W_hid_to_hid=np.eye(n_rec_units, dtype=np.float32) * 0.9,
            backwards=True
    )

    # first combine the forward and backward recurrent layers...
    net = lnn.layers.ConcatLayer([fwd, bck], name='fwd + bck', axis=-1)
    net = lnn.layers.DropoutLayer(net, p=0.5)

    crf = spg.layers.CrfLayer(net, mask_input=mask_in,
                              num_states=out_size, name='CRF')

    return crf


def compute_loss(network, target, mask):
    loss = spg.objectives.neg_log_likelihood(network, target, mask)
    return lnn.objectives.aggregate(loss, mode='mean')


def build_net(feature_shape, batch_size, max_seq_len, out_size):
    # create the network
    feature_var = tt.tensor3('feature_input', dtype='float32')
    target_var = tt.tensor3('target_output', dtype='float32')
    mask_var = tt.matrix('mask_input', dtype='float32')

    network = stack_layers(feature_var, mask_var,
                           feature_shape, batch_size, max_seq_len, out_size)

    # create train function
    weight_decay = lnn.regularization.regularize_network_params(
        network, lnn.regularization.l2) * 1e-6
    loss = compute_loss(network, target_var, mask_var) + weight_decay

    params = lnn.layers.get_all_params(network, trainable=True)
    updates = lnn.updates.adam(loss, params, learning_rate=0.001)
    train = theano.function([feature_var, mask_var, target_var], loss,
                            updates=updates)

    # create test and process function. process just computes the prediction
    # without computing the loss, and thus does not need target labels
    test_loss = compute_loss(network, target_var, mask_var) + weight_decay
    test = theano.function([feature_var, mask_var, target_var], test_loss)

    viterbi_out = lnn.layers.get_output(network, mode='viterbi')
    process = theano.function([feature_var, mask_var], viterbi_out)

    return nn.NeuralNetwork(network, train, test, process)


BATCH_SIZE = 64
MAX_SEQ_LEN = 1024


def main():

    print(Colors.red('Loading data...\n'))

    feature_computer = data.LogFiltSpec()
    # load all data sets
    train_set, val_set, test_set, gt_files = data.load_datasets(
        preprocessors=[dmgr.preprocessing.DataWhitener(),
                       dmgr.preprocessing.MaxNorm()],
        compute_features=feature_computer
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
            batch_size=BATCH_SIZE,
            max_seq_len=MAX_SEQ_LEN,
            out_size=train_set.target_shape[0]
    )

    print(Colors.blue('Neural Network:'))
    print(train_neural_net)
    print('')

    print(Colors.red('Starting training...\n'))

    best_params = nn.train(
            train_neural_net, train_set, n_epochs=1000, batch_size=BATCH_SIZE,
            validation_set=val_set, early_stop=20,
            batch_iterator=dmgr.iterators.iterate_datasources,
            sequence_length=MAX_SEQ_LEN
    )

    print(Colors.red('\nStarting testing...\n'))

    del train_neural_net

    # build test rnn with batch size 1 and no max sequence length
    test_neural_net = build_net(
            feature_shape=test_set.feature_shape,
            batch_size=1,
            max_seq_len=None,
            out_size=test_set.target_shape[0]
    )

    test_neural_net.set_parameters(best_params)

    dest_dir = os.path.join('results', os.path.splitext(__file__)[0])
    pred_files = test.compute_labeling(test_neural_net, test_set,
                                       dest_dir=dest_dir,
                                       rnn=True, out_onehot=False)
    print('\tWrote chord predictions to {}.'.format(dest_dir))

    print(Colors.red('\nResults:\n'))

    test_gt_files = dmgr.files.match_files(
            pred_files, gt_files, test.PREDICTION_EXT, data.GT_EXT
    )

    test.print_scores(test.compute_average_scores(test_gt_files, pred_files))

    print('')


if __name__ == '__main__':
    main()
