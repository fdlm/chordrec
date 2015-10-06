from __future__ import print_function
import theano
import theano.tensor as tt
import lasagne as lnn

import nn
import dmgr
import test
import data

from nn.utils import Colors


def stack_layers(feature_var, feature_shape, batch_size, out_size):
    net = lnn.layers.InputLayer(name='input',
                                shape=(batch_size,) + feature_shape,
                                input_var=feature_var)

    nl = lnn.nonlinearities.rectify
    num_units = 64

    net = lnn.layers.DenseLayer(net, num_units=num_units, nonlinearity=nl)
    net = lnn.layers.DropoutLayer(net, p=0.5)
    net = lnn.layers.DenseLayer(net, num_units=num_units, nonlinearity=nl)
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

    l2_penalty = lnn.regularization.regularize_network_params(
        network, lnn.regularization.l2) * 1e-4
    loss = compute_loss(prediction, target_var) + l2_penalty

    params = lnn.layers.get_all_params(network, trainable=True)
    updates = lnn.updates.adam(loss, params, learning_rate=0.0001)

    # max norm constraint on weights
    all_non_bias_params = lnn.layers.get_all_params(network, trainable=True,
                                                    regularizable=True)
    # for param, update in updates.iteritems():
    #     if param in all_non_bias_params:
    #         updates[param] = lnn.updates.norm_constraint(update, max_norm=4)

    train = theano.function([feature_var, target_var], loss,
                            updates=updates)

    # create test and process function. process just computes the prediction
    # without computing the loss, and thus does not need target labels
    test_prediction = lnn.layers.get_output(network, deterministic=True)
    test_loss = compute_loss(test_prediction, target_var) + l2_penalty

    test = theano.function([feature_var, target_var], test_loss)
    process = theano.function([feature_var], test_prediction)

    return nn.NeuralNetwork(network, train, test, process)


BATCH_SIZE = 1024


def main():

    print(Colors.red('Loading data...\n'))

    mirex09 = data.load_mirex09_dataset()
    robbie = data.load_robbie_dataset()
    files = data.combine_files(
        mirex09.get_rand_split(),
        robbie.get_rand_split(val_perc=0., test_perc=0.)
    )

    train_set, val_set, test_set = data.get_preprocessed_context_datasources(
        files, context_size=5,
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
        threaded=10
    )

    print(Colors.red('\nStarting testing...\n'))

    dest_dir = './results/dnn'
    neural_net.set_parameters(best_params)
    pred_files = test.compute_labeling(neural_net, test_set, dest_dir=dest_dir,
                                       rnn=False)
    print('\tWrote chord predictions to {}.'.format(dest_dir))

    print(Colors.red('\nResults:\n'))

    test_gt_files = dmgr.files.match_files(
        pred_files, mirex09.gt_files, test.PREDICTION_EXT, data.GT_EXT
    )

    test.print_scores(test.compute_average_scores(test_gt_files, pred_files))

    print('')


if __name__ == '__main__':
    main()
