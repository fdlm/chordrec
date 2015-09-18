from __future__ import print_function
import theano
import theano.tensor as tt
import sklearn.metrics
import lasagne as lnn

import nn
import data

from nn.utils import Colors


N_SPECTRA = 3


def stack_layers(feature_var, feature_shape, batch_size, out_size):
    net = lnn.layers.InputLayer(name='input',
                                shape=(batch_size,) + feature_shape,
                                input_var=feature_var)

    # first, we have to reshape the input in a way that the spectra for
    # each frame_size are 'color channels', the time context is 'rows', and
    # the spectrogram bins are 'columns', i.e. the shape has to be
    # (None, frame_size, context, bins)

    net = lnn.layers.reshape(net, (-1,  # we do not know how many inputs we get
                                   feature_shape[0],  # context
                                   N_SPECTRA,  # number of spectra
                                   feature_shape[1] / N_SPECTRA  # bins per spec
                                   ),
                             name='reshape'
                             )

    # now, shuffle dims so that number of spectra is in the second dim
    net = lnn.layers.dimshuffle(net, (0, 2, 1, 3), name='transpose')

    net = lnn.layers.conv.Conv2DLayer(net, num_filters=25,
                                      filter_size=(3, 12),
                                      name='conv')

    net = lnn.layers.DropoutLayer(net, p=0.3)

    net = lnn.layers.DenseLayer(net, num_units=128,
                                nonlinearity=lnn.nonlinearities.rectify,
                                name='fc-1')

    net = lnn.layers.DenseLayer(net, num_units=128,
                                nonlinearity=lnn.nonlinearities.rectify,
                                name='fc-2')

    net = lnn.layers.DropoutLayer(net, p=0.3)

    # output layer
    net = lnn.layers.DenseLayer(net, name='output', num_units=out_size,
                                nonlinearity=lnn.nonlinearities.softmax)

    return net


def compute_loss(prediction, target):
    return lnn.objectives.categorical_crossentropy(prediction, target).mean()


def build_net(feature_shape, batch_size, out_size):
    # create the network
    feature_var = tt.tensor3('feature_input', dtype='float32')
    target_var = tt.matrix('target_output', dtype='int32')
    network = stack_layers(feature_var, feature_shape, batch_size, out_size)

    # create train function
    prediction = lnn.layers.get_output(network)
    loss = compute_loss(prediction, target_var)
    params = lnn.layers.get_all_params(network, trainable=True)

    updates = lnn.updates.adam(loss, params, learning_rate=0.00002)

    # max norm constraint on weights
    all_non_bias_params = lnn.layers.get_all_params(network, trainable=True,
                                                    regularizable=True)
    for param, update in updates.iteritems():
        if param in all_non_bias_params:
            updates[param] = lnn.updates.norm_constraint(update, max_norm=1.)

    train = theano.function([feature_var, target_var], loss,
                            updates=updates)

    # create test and process function. process just computes the prediction
    # without computing the loss, and thus does not need target labels
    test_prediction = lnn.layers.get_output(network, deterministic=True)
    test_loss = compute_loss(test_prediction, target_var)

    test = theano.function([feature_var, target_var],
                           test_loss)
    process = theano.function([feature_var], test_prediction)

    return nn.NeuralNetwork(network, train, test, process)


BATCH_SIZE = 1024


def main():

    print(Colors.red('Loading data...\n'))

    beatles = data.Beatles()
    files = beatles.get_fold_split()
    train_set, val_set, test_set = data.get_whitened_context_datasources(files)

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
        out_size=train_set.target_shape[0]
    )

    print(Colors.blue('Neural Network:'))
    print(neural_net)
    print('')

    print(Colors.red('Starting training...\n'))

    best_params = nn.train(
        neural_net, train_set, n_epochs=100, batch_size=BATCH_SIZE,
        validation_set=val_set, early_stop=10,
        threaded=5
    )

    print(Colors.red('Starting testing...\n'))

    predictions = nn.predict(
        neural_net, test_set, BATCH_SIZE
    )

    pred_class = predictions.argmax(axis=1)
    ids = range(test_set.n_data)
    correct_class = test_set[ids][1].argmax(axis=1)

    print(sklearn.metrics.classification_report(correct_class, pred_class))


if __name__ == '__main__':
    main()
