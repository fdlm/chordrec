from __future__ import print_function
import numpy as np
import theano
import theano.tensor as tt
import sklearn.metrics
import lasagne as lnn

import nn
import data
import dmgr

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

    net = lnn.layers.RecurrentLayer(
        net, name='recurrent', num_units=128, mask_input=mask_in,
        grad_clipping=1., W_in_to_hid=lnn.init.HeUniform(),
        # W_hid_to_hid=np.eye(128, dtype=np.float32) * 0.9
        W_hid_to_hid=lnn.init.HeUniform()
    )

    # In order to connect a recurrent layer to a dense layer, we need to
    # flatten the first two dimensions (our "sample dimensions"); this will
    # cause each time step of each sequence to be processed independently
    net = lnn.layers.ReshapeLayer(net, (-1, 128), name='reshape to single')
    net = lnn.layers.DropoutLayer(net, p=0.5)

    net = lnn.layers.DenseLayer(net, num_units=128,
                                nonlinearity=lnn.nonlinearities.rectify,
                                name='fc-1')
    net = lnn.layers.DropoutLayer(net, p=0.5)

    net = lnn.layers.DenseLayer(net, num_units=out_size,
                                nonlinearity=lnn.nonlinearities.softmax,
                                name='output')
    # To reshape back to our original shape, we can use the symbolic shape
    # variables we retrieved above.
    net = lnn.layers.ReshapeLayer(net,
                                  (true_batch_size, true_seq_len, out_size),
                                  name='output-reshape')

    return net


def compute_loss(prediction, target):
    return lnn.objectives.categorical_crossentropy(prediction, target).mean()


def build_net(feature_shape, batch_size, max_seq_len, out_size):
    # create the network
    feature_var = tt.tensor3('feature_input', dtype='float32')
    target_var = tt.tensor3('target_output', dtype='int32')
    mask_var = tt.matrix('mask_input', dtype='float32')

    network = stack_layers(feature_var, mask_var,
                           feature_shape, batch_size, max_seq_len, out_size)

    # create train function
    prediction = lnn.layers.get_output(network)

    l2_penalty = lnn.regularization.regularize_network_params(
        network, lnn.regularization.l2)
    loss = compute_loss(prediction, target_var) + l2_penalty * 1e-6

    params = lnn.layers.get_all_params(network, trainable=True)

    updates = lnn.updates.adam(loss, params, learning_rate=0.001)

    train = theano.function([feature_var, mask_var, target_var], loss,
                            updates=updates)

    # create test and process function. process just computes the prediction
    # without computing the loss, and thus does not need target labels
    test_prediction = lnn.layers.get_output(network, deterministic=True)
    test_loss = compute_loss(test_prediction, target_var)

    test = theano.function([feature_var, mask_var, target_var],
                           test_loss)
    process = theano.function([feature_var, mask_var], test_prediction)

    return nn.NeuralNetwork(network, train, test, process)


BATCH_SIZE = 64
MAX_SEQ_LEN = 1024


def main():

    print(Colors.red('Loading data...\n'))

    beatles = data.Beatles()
    files = beatles.get_fold_split()

    train_set, val_set, test_set = data.get_preprocessed_datasources(
        files,
        preprocessors=[dmgr.preprocessing.DataWhitener(),
                       dmgr.preprocessing.MaxNorm()]
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

    print(Colors.red('Starting testing...\n'))

    del train_neural_net

    test_neural_net = build_net(
        feature_shape=test_set.feature_shape,
        batch_size=1,
        max_seq_len=None,
        out_size=test_set.target_shape[0]
    )

    test_neural_net.set_parameters(best_params)

    # when predicting, we feed the net one song at a time
    predictions = nn.predict_rnn(
        test_neural_net, test_set, batch_size=1,
        batch_iterator=dmgr.iterators.iterate_datasources,
        sequence_length=None
    )

    pred_class = predictions.argmax(axis=1)
    ids = range(test_set.n_data)
    correct_class = test_set[ids][1].argmax(axis=1)

    print(sklearn.metrics.classification_report(correct_class, pred_class))


if __name__ == '__main__':
    main()
