from __future__ import print_function
import os
import theano
import theano.tensor as tt
import lasagne as lnn

import nn
import data
import dmgr
import test

from nn.utils import Colors


def stack_layers(feature_var, feature_shape, batch_size, out_size):
    net = lnn.layers.InputLayer(name='input',
                                shape=(batch_size,) + feature_shape,
                                input_var=feature_var)

    nl = lnn.nonlinearities.rectify

    net = lnn.layers.DenseLayer(net, num_units=100, nonlinearity=nl,
                                name='fe-1')
    net = lnn.layers.DropoutLayer(net, p=0.3)
    net = lnn.layers.DenseLayer(net, num_units=100, nonlinearity=nl,
                                name='fe-2')
    net = lnn.layers.DropoutLayer(net, p=0.3)
    net = lnn.layers.DenseLayer(net, num_units=100, nonlinearity=nl,
                                name='fe-3')
    net = lnn.layers.DropoutLayer(net, p=0.3)

    # "feature extraction" output layer
    fe_out = lnn.layers.DenseLayer(net, name='fe-out', num_units=out_size,
                                   nonlinearity=lnn.nonlinearities.softmax)

    net = lnn.layers.DenseLayer(net, num_units=100, nonlinearity=nl,
                                name='dnn-1')
    net = lnn.layers.DropoutLayer(net, p=0.3)
    net = lnn.layers.DenseLayer(net, num_units=100, nonlinearity=nl,
                                name='dnn-2')
    net = lnn.layers.DropoutLayer(net, p=0.3)
    net = lnn.layers.DenseLayer(net, num_units=100, nonlinearity=nl,
                                name='dnn-3')
    net = lnn.layers.DropoutLayer(net, p=0.3)

    # output layer
    net = lnn.layers.DenseLayer(net, name='output', num_units=out_size,
                                nonlinearity=lnn.nonlinearities.softmax)

    return net, fe_out


def compute_loss(prediction, target):
    return lnn.objectives.categorical_crossentropy(prediction, target).mean()


def build_net(feature_shape, batch_size, out_size):
    # create the network
    feature_var = tt.matrix('feature_input', dtype='float32')
    target_var = tt.matrix('target_output', dtype='float32')
    net, fe_net = stack_layers(feature_var, feature_shape, batch_size, out_size)

    # create feature extraction train function
    fe_pred = lnn.layers.get_output(fe_net)
    fe_loss = compute_loss(fe_pred, target_var)
    fe_params = lnn.layers.get_all_params(fe_net, trainable=True)
    fe_upd = lnn.updates.adadelta(fe_loss, fe_params)

    train_fe = theano.function([feature_var, target_var],
                               fe_loss, updates=fe_upd)

    fe_test_pred = lnn.layers.get_output(fe_net, deterministic=True)
    fe_test_loss = compute_loss(fe_test_pred, target_var)
    test_fe = theano.function([feature_var, target_var], fe_test_loss)

    # create dnn train function
    pred = lnn.layers.get_output(net)
    loss = compute_loss(pred, target_var)
    # get only non-fe params
    params = list(set(lnn.layers.get_all_params(net)) - set(fe_params))
    upd = lnn.updates.adadelta(loss, params)

    train = theano.function([feature_var, target_var],
                            loss, updates=upd)

    # create test and process function. process just computes the prediction
    # without computing the loss, and thus does not need target labels
    test_prediction = lnn.layers.get_output(net, deterministic=True)
    test_loss = compute_loss(test_prediction, target_var)

    test = theano.function([feature_var, target_var], test_loss)
    process = theano.function([feature_var], test_prediction)

    return (nn.NeuralNetwork(fe_net, train_fe, test_fe, None),
            nn.NeuralNetwork(net, train, test, process))


BATCH_SIZE = 512


def main():

    print(Colors.red('Loading data...\n'))

    feature_computer = data.ConstantQ()
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

    fe_net, neural_net = build_net(
        feature_shape=train_set.feature_shape,
        batch_size=None,
        out_size=train_set.target_shape[0]
    )

    print(Colors.blue('Neural Network:'))
    print(neural_net)
    print('')

    print(Colors.red('Starting feature extraction training...\n'))

    best_fe_params = nn.train(
        fe_net, train_set, n_epochs=500, batch_size=BATCH_SIZE,
        validation_set=val_set, early_stop=20,
        threaded=10
    )

    print(Colors.red('Starting neural network training...\n'))

    best_params = nn.train(
        neural_net, train_set, n_epochs=500, batch_size=BATCH_SIZE,
        validation_set=val_set, early_stop=20,
        threaded=10
    )

    print(Colors.red('\nStarting testing...\n'))

    dest_dir = os.path.join('results', os.path.splitext(__file__)[0])
    # neural_net.set_parameters(best_params)
    pred_files = test.compute_labeling(neural_net, test_set, dest_dir=dest_dir,
                                       fps=feature_computer.fps, rnn=False)
    print('\tWrote chord predictions to {}.'.format(dest_dir))

    print(Colors.red('\nResults:\n'))

    test_gt_files = dmgr.files.match_files(
            pred_files, gt_files,
            test.PREDICTION_EXT, data.GT_EXT
    )

    test.print_scores(test.compute_average_scores(test_gt_files, pred_files))

    print('')


if __name__ == '__main__':
    main()

