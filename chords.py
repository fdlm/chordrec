from __future__ import print_function
import theano
import theano.tensor as tt
import sklearn.metrics
import lasagne as lnn

import nn
import dmgr
import data

from nn.utils import Colors


def stack_layers(feature_var, feature_shape, batch_size, out_size):
    net = lnn.layers.InputLayer(name='input',
                                shape=(batch_size,) + feature_shape,
                                input_var=feature_var)

    nl = lnn.nonlinearities.rectify

    net = lnn.layers.DenseLayer(net, num_units=512, nonlinearity=nl)
    net = lnn.layers.DropoutLayer(net, p=0.5)
    net = lnn.layers.DenseLayer(net, num_units=256, nonlinearity=nl)
    net = lnn.layers.DropoutLayer(net, p=0.5)
    net = lnn.layers.DenseLayer(net, num_units=256, nonlinearity=nl)
    net = lnn.layers.DropoutLayer(net, p=0.5)

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

    updates = lnn.updates.adam(loss, params, learning_rate=0.00001)

    # max norm constraint on weights
    all_non_bias_params = lnn.layers.get_all_params(network, regularizable=True)
    for param, update in updates.iteritems():
        if param in all_non_bias_params:
            updates[param] = lnn.updates.norm_constraint(update, max_norm=4.)

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

    train_set = dmgr.datasources.AggregatedDataSource.from_files(
        files['train']['feat'], files['train']['targ'], memory_mapped=True,
        data_source_type=dmgr.datasources.ContextDataSource,
        context_size=3
    )

    val_set = dmgr.datasources.AggregatedDataSource.from_files(
        files['val']['feat'], files['val']['targ'], memory_mapped=True,
        data_source_type=dmgr.datasources.ContextDataSource,
        context_size=3
    )

    test_set = dmgr.datasources.AggregatedDataSource.from_files(
        files['test']['feat'], files['test']['targ'], memory_mapped=True,
        data_source_type=dmgr.datasources.ContextDataSource,
        context_size=3
    )

    preproc = dmgr.preprocessing.DataWhitener()
    preproc.train(train_set, batch_size=4096)

    train_set = dmgr.datasources.PreProcessedDataSource(
        train_set, preproc
    )

    val_set = dmgr.datasources.PreProcessedDataSource(
        val_set, preproc
    )

    test_set = dmgr.datasources.PreProcessedDataSource(
        test_set, preproc
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
        out_size=train_set.target_shape[0]
    )

    print(Colors.blue('Neural Network:'))
    print(neural_net)
    print('')

    print(Colors.red('Starting training...\n'))

    best_params = nn.train(
        neural_net, train_set, n_epochs=100, batch_size=BATCH_SIZE,
        validation_set=val_set, early_stop=10
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
