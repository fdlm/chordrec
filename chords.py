from __future__ import print_function
import theano
import theano.tensor as tt
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

    net = lnn.layers.DropoutLayer(net, p=0.3)
    net = lnn.layers.DenseLayer(net, num_units=512, nonlinearity=nl)
    net = lnn.layers.DropoutLayer(net, p=0.3)
    net = lnn.layers.DenseLayer(net, num_units=512, nonlinearity=nl)
    net = lnn.layers.DropoutLayer(net, p=0.3)
    net = lnn.layers.DenseLayer(net, num_units=512, nonlinearity=nl)

    # output layer
    net = lnn.layers.DenseLayer(net, name='output', num_units=out_size,
                                nonlinearity=lnn.nonlinearities.softmax)

    return net


def compute_loss(prediction, target):
    return lnn.objectives.categorical_crossentropy(prediction, target).mean()


def build_net(feature_shape, batch_size, out_size):
    # create the network
    feature_var = tt.matrix('feature_input', dtype='float32')
    target_var = tt.matrix('target_output', dtype='int32')
    network = stack_layers(feature_var, feature_shape, batch_size, out_size)

    # create train function
    prediction = lnn.layers.get_output(network)
    loss = compute_loss(prediction, target_var)
    params = lnn.layers.get_all_params(network, trainable=True)

    updates = lnn.updates.adadelta(loss, params)

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
    # load data
    data_dir = 'data/beatles'
    src_ext = '.flac'
    gt_ext = '.chords'
    dst_dir = 'feature_cache/beatles'

    val_def = data_dir + '/splits/8-fold_cv_album_distributed_0.fold'
    test_def = data_dir + '/splits/8-fold_cv_album_distributed_1.fold'
    file_filter = '*'

    print(Colors.red('Loading data...\n'))

    src_files = dmgr.files.expand(data_dir, file_filter + src_ext)
    gt_files = dmgr.files.expand(data_dir, file_filter + gt_ext)

    src_train_files, src_val_files, src_test_files = \
        dmgr.files.predefined_train_val_test_split(
            src_files, val_def, test_def
        )

    gt_train_files = dmgr.files.match_files(src_train_files, gt_files,
                                            src_ext, gt_ext)
    gt_val_files = dmgr.files.match_files(src_val_files, gt_files,
                                          src_ext, gt_ext)
    gt_test_files = dmgr.files.match_files(src_test_files, gt_files,
                                           src_ext, gt_ext)

    feat_train_files, target_train_files = dmgr.files.prepare(
        src_train_files, gt_train_files, dst_dir,
        compute_feat=data.compute_features,
        compute_targets=data.compute_targets,
        fps=data.FPS
    )

    feat_val_files, target_val_files = dmgr.files.prepare(
        src_val_files, gt_val_files, dst_dir,
        compute_feat=data.compute_features,
        compute_targets=data.compute_targets,
        fps=data.FPS
    )

    feat_test_files, target_test_files = dmgr.files.prepare(
        src_test_files, gt_test_files, dst_dir,
        compute_feat=data.compute_features,
        compute_targets=data.compute_targets,
        fps=data.FPS
    )

    train_set = dmgr.datasources.AggregatedDataSource.from_files(
        feat_train_files, target_train_files, memory_mapped=True
    )

    val_set = dmgr.datasources.AggregatedDataSource.from_files(
        feat_val_files, target_val_files, memory_mapped=True
    )

    test_set = dmgr.datasources.AggregatedDataSource.from_files(
        feat_test_files, target_test_files, memory_mapped=True
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
    print('\t', test_set)
    print('')

    # build network
    print(Colors.red('Building network...\n'))

    train_network = build_net(
        feature_shape=train_set.feature_shape,
        batch_size=BATCH_SIZE,
        out_size=train_set.target_shape[0]
    )

    print(Colors.blue('Neural Network:'))
    print(train_network)
    print('')

    print(Colors.red('Starting training...\n'))

    best_params = nn.train(
        train_network, train_set, n_epochs=100, batch_size=BATCH_SIZE,
        validation_set=val_set, early_stop=10
    )

    print(Colors.red('Starting testing...\n'))


if __name__ == '__main__':
    main()
