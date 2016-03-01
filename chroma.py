from __future__ import print_function
import os
import collections
import theano
import theano.tensor as tt
import lasagne as lnn
import yaml
import numpy as np
from sacred import Experiment

import nn
import dmgr
from nn.utils import Colors

import test
import data
import features
import targets
from exp_utils import (PickleAndSymlinkObserver, TempDir, create_optimiser,
                       ParamSaver)
import dnn
import convnet


# Initialise Sacred experiment
ex = Experiment('Deep Neural Network / Chroma Target')
ex.observers.append(PickleAndSymlinkObserver())
data.add_sacred_config(ex)
features.add_sacred_config(ex)
targets.add_sacred_config(ex)


def compute_chroma_loss(prediction, target):
    # need to clip predictions for numerical stability
    eps = 1e-7
    pred_clip = tt.clip(prediction, eps, 1.-eps)
    return lnn.objectives.binary_crossentropy(pred_clip, target).mean()


def build_net(feature_shape, optimiser, out_size_chroma, out_size_chords,
              batch_norm, conv, dense, l2):

    # input variables
    feature_var = (tt.tensor3('feature_input', dtype='float32')
                   if len(feature_shape) > 1 else
                   tt.matrix('feature_input', dtype='float32'))
    chroma_var = tt.matrix('chroma_output', dtype='float32')
    chord_var = tt.matrix('chord_output', dtype='float32')

    # stack more layers
    net = lnn.layers.InputLayer(name='input',
                                shape=(None,) + feature_shape,
                                input_var=feature_var)

    if conv:
        # reshape to 1 "color" channel
        net = lnn.layers.reshape(net, shape=(-1, 1) + feature_shape,
                                 name='reshape')
        net = convnet.stack_layers(net, batch_norm, conv)

    net = dnn.stack_layers(net, batch_norm, **dense)

    # output layers
    chrm = lnn.layers.DenseLayer(
        net, name='chroma_out', num_units=out_size_chroma,
        nonlinearity=lnn.nonlinearities.sigmoid)

    crds = lnn.layers.DenseLayer(
        chrm, name='chords', num_units=out_size_chords,
        nonlinearity=lnn.nonlinearities.softmax)

    # tag chord classification parameters so we can distinguish them later
    for p in crds.get_params():
        crds.params[p].add('chord')

    # ====================================== Theano functions for chroma target
    # trains all the network parameters until chroma output

    # create chroma train function
    chrm_prediction = lnn.layers.get_output(chrm)
    l2_chrm = lnn.regularization.regularize_network_params(
        chrm, lnn.regularization.l2) * l2
    chrm_loss = compute_chroma_loss(chrm_prediction, chroma_var) + l2_chrm
    chrm_params = lnn.layers.get_all_params(chrm, trainable=True)
    chrm_updates = optimiser(chrm_loss, chrm_params)
    train_chroma = theano.function(
        [feature_var, chroma_var], chrm_loss, updates=chrm_updates)

    # create chroma test and process function. process just computes the
    # prediction without computing the loss, and thus does not need
    # target labels
    chrm_test_prediction = lnn.layers.get_output(chrm, deterministic=True)
    chrm_test_loss = compute_chroma_loss(
        chrm_test_prediction, chroma_var) + l2_chrm
    test_chroma = theano.function([feature_var, chroma_var],
                                  [chrm_test_loss, chrm_test_prediction])
    process_chroma = theano.function([feature_var], chrm_test_prediction)

    # =============================== Theano functions for chord classification
    # takes the chroma target as input and does linear regression on chord
    # target (softmax layer). Only trains the parameters of the last
    # layer (tagged as 'chord' in the code above)

    # create chord classification train function
    crds_prediction = lnn.layers.get_output(crds)
    l2_crds = lnn.regularization.regularize_network_params(
        crds, lnn.regularization.l2,
        tags={'regularizable': True, 'chord': True}) * l2
    crds_loss = dnn.compute_loss(crds_prediction, chord_var) + l2_crds
    crds_params = lnn.layers.get_all_params(crds, trainable=True, chord=True)
    crds_updates = optimiser(crds_loss, crds_params)
    train_chords = theano.function(
        [feature_var, chord_var], crds_loss, updates=crds_updates
    )

    # create chord test and process function. process just computes the
    # prediction without computing the loss, and thus does not need
    # target labels
    crds_test_prediction = lnn.layers.get_output(crds, deterministic=True)
    crds_test_loss = dnn.compute_loss(
        crds_test_prediction, chord_var) + l2_chrm
    test_chords = theano.function([feature_var, chord_var],
                                  [crds_test_loss, crds_test_prediction])
    process_chords = theano.function([feature_var], crds_test_prediction)

    # Create the neural network classes and return them
    chroma_net = nn.NeuralNetwork(
        chrm, train_chroma, test_chroma, process_chroma
    )

    chords_net = nn.NeuralNetwork(
        crds, train_chords, test_chords, process_chords
    )

    return chroma_net, chords_net


def compute_chroma(network, agg_dataset, dest_dir, extension='.chroma.npy'):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    chroma_files = []

    for ds_idx in range(agg_dataset.n_datasources):
        ds = agg_dataset.get_datasource(ds_idx)

        # skip targets
        data, _ = ds[:]
        chromas = network.process(data)
        chroma_file = os.path.join(dest_dir, ds.name + extension)
        np.save(chroma_file, chromas)
        chroma_files.append(chroma_file)

    return chroma_files


@ex.config
def config():
    observations = 'results'

    datasource = dict(
        context_size=7,
    )

    feature_extractor = None

    target = None

    net = dict(
        conv={},
        dense=dict(
            num_layers=3,
            num_units=256,
            dropout=0.5,
            nonlinearity='rectify',
        ),
        l2=1e-4,
        batch_norm=False,
    )

    optimiser = dict(
        name='adam',
        params=dict(
            learning_rate=0.0001
        ),
        schedule=None
    )

    training = dict(
        num_epochs=500,
        early_stop=20,
        batch_size=512,
        early_stop_acc=True,
    )

    testing = dict(
        test_on_val=False
    )


@ex.named_config
def no_context():
    datasource = dict(
        context_size=0
    )

    net = dict(
        num_units=100,
        dropout=0.3,
        l2_lambda=0.
    )


@ex.automain
def main(_config, _run, observations, datasource, net, feature_extractor,
         target, optimiser, training):

    if feature_extractor is None:
        print(Colors.red('ERROR: Specify a feature extractor!'))
        return 1

    if target is not None:
        print(Colors.red('WARNING: Selected target ignored.'))

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

            target_chroma = targets.ChromaTarget(
                feature_extractor['params']['fps'])

            target_chords = targets.create_target(
                feature_extractor['params']['fps'],
                target
            )

            feature_ext = features.create_extractor(feature_extractor)

            train_set, val_set, test_set, gt_files = data.create_datasources(
                dataset_names=datasource['datasets'],
                preprocessors=datasource['preprocessors'],
                compute_features=feature_ext,
                compute_targets=target_chroma,
                context_size=datasource['context_size'],
                test_fold=test_fold,
                val_fold=val_fold,
                cached=datasource['cached']
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

            opt, learn_rate = create_optimiser(optimiser)
            chroma_net, chords_net = build_net(
                feature_shape=train_set.feature_shape,
                optimiser=opt,
                out_size_chroma=train_set.target_shape[0],
                out_size_chords=target_chords.num_classes,
                **net
            )

            print(Colors.blue('Chroma Network:'))
            print(chroma_net)
            print('')

            print(Colors.blue('Chords Network:'))
            print(chords_net)
            print('')

            print(Colors.red('Starting training chroma network...\n'))

            updates = []
            if optimiser['schedule'] is not None:
                updates.append(
                    nn.LearnRateSchedule(
                        learn_rate=learn_rate, **optimiser['schedule'])
                )

            best_params, train_losses, val_losses = nn.train(
                chroma_net, train_set, n_epochs=training['num_epochs'],
                batch_size=training['batch_size'], validation_set=val_set,
                early_stop=training['early_stop'],
                early_stop_acc=training['early_stop_acc'],
                threaded=10,
                updates=updates,
            )

            # we need to create a new dataset with a new target (chords)
            del train_set
            del val_set
            del test_set
            del gt_files

            train_set, val_set, test_set, gt_files = data.create_datasources(
                dataset_names=datasource['datasets'],
                preprocessors=datasource['preprocessors'],
                compute_features=feature_ext,
                compute_targets=target_chords,
                context_size=datasource['context_size'],
                test_fold=datasource['test_fold'],
                val_fold=datasource['val_fold'],
                cached=datasource['cached']
            )

            print(Colors.blue('Train Set:'))
            print('\t', train_set)

            print(Colors.blue('Validation Set:'))
            print('\t', val_set)

            print(Colors.blue('Test Set:'))
            print('\t', test_set)
            print('')

            print(Colors.red('Starting training chord network...\n'))

            best_params, train_losses, val_losses = nn.train(
                chords_net, train_set, n_epochs=training['num_epochs'],
                batch_size=training['batch_size'], validation_set=val_set,
                early_stop=training['early_stop'],
                early_stop_acc=training['early_stop_acc'],
                threaded=10,
            )

            print(Colors.red('\nStarting testing...\n'))

            param_file = os.path.join(
                exp_dir, 'params_fold_{}.pkl'.format(test_fold))
            chords_net.save_parameters(param_file)
            ex.add_artifact(param_file)

            pred_files = test.compute_labeling(
                chords_net, target_chords, test_set, dest_dir=exp_dir,
                rnn=False
            )

            # compute chroma vectors for the test set
            for cf in compute_chroma(chroma_net, test_set, dest_dir=exp_dir):
                ex.add_artifact(cf)

            test_gt_files = dmgr.files.match_files(
                pred_files, gt_files, test.PREDICTION_EXT, data.GT_EXT
            )

            all_pred_files += pred_files
            all_gt_files += test_gt_files

            print(Colors.blue('Results:'))
            scores = test.compute_average_scores(test_gt_files, pred_files)
            # convert to float so yaml output looks nice
            for k in scores:
                scores[k] = float(scores[k])
            test.print_scores(scores)

            result_file = os.path.join(
                exp_dir, 'results_fold_{}.yaml'.format(test_fold))
            yaml.dump(dict(scores=scores,
                           train_losses=map(float, train_losses),
                           val_losses=map(float, val_losses)),
                      open(result_file, 'w'))
            ex.add_artifact(result_file)

        # if there is something to aggregate
        if len(datasource['test_fold']) > 1:
            print(Colors.yellow('\nAggregated Results:\n'))
            scores = test.compute_average_scores(all_gt_files, all_pred_files)
            # convert to float so yaml output looks nice
            for k in scores:
                scores[k] = float(scores[k])
            test.print_scores(scores)
            result_file = os.path.join(exp_dir, 'results.yaml')
            yaml.dump(dict(scores=scores), open(result_file, 'w'))
            ex.add_artifact(result_file)

        for pf in all_pred_files:
            ex.add_artifact(pf)

    print(Colors.magenta('Stopping experiment ' + ex.observers[0].hash()))
