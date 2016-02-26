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
import targets
import dnn
from plotting import CrfPlotter
from exp_utils import (PickleAndSymlinkObserver, TempDir, create_optimiser,
                       ParamSaver)

# Initialise Sacred experiment
ex = Experiment('Conditional Random Field')
ex.observers.append(PickleAndSymlinkObserver())
data.add_sacred_config(ex)
features.add_sacred_config(ex)
targets.add_sacred_config(ex)


def compute_loss(network, target, mask):
    loss = spg.objectives.neg_log_likelihood(network, target, mask)
    loss /= mask.sum(axis=1)  # normalise to sequence length
    return lnn.objectives.aggregate(loss, mode='mean')


def dnn_loss(prediction, target, mask):
    # need to clip predictions for numerical stability
    eps = 1e-7
    pred_clip = tt.clip(prediction, eps, 1.-eps)
    loss = lnn.objectives.categorical_crossentropy(pred_clip, target)
    return lnn.objectives.aggregate(loss, mask, mode='normalized_sum')


def add_dense_layers(net, feature_shape, true_batch_size, true_seq_len,
                     num_units, num_layers, dropout):

    net = lnn.layers.ReshapeLayer(net, (-1,) + feature_shape,
                                  name='reshape to single')

    for i in range(num_layers):
        net = lnn.layers.DenseLayer(net, num_units=num_units,
                                    name='fc-{}'.format(i))

        if dropout > 0.0:
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


def build_dense_ip(feature_shape, net, optimiser, out_size):
    ip_net = dnn.build_net(feature_shape, batch_size=None,
                           l2=net['l2_lambda'],
                           num_units=net['num_units'],
                           num_layers=net['num_layers'],
                           dropout=net['dropout'],
                           batch_norm=net['batch_norm'],
                           nonlinearity=net['nonlinearity'],
                           optimiser=optimiser,
                           out_size=out_size)

    return ip_net


def build_net(feature_shape, batch_size, l2_lambda, max_seq_len,
              dense, recurrent, init_softmax, optimiser, out_size,
              input_processor, input_processor_params):

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

    true_batch_size, true_seq_len = feature_var.shape[:2]

    if input_processor is not None:
        net = input_processor(net, true_batch_size, true_seq_len)
        # leave out softmax parameters
        lnn.layers.set_all_param_values(net, input_processor_params[:-2])

    # add dense layers between input and crf
    if dense['num_layers'] > 0:
        net = add_dense_layers(
            net, feature_shape, true_batch_size, true_seq_len, **dense
        )

    # add recurrent layers between input and crf
    if recurrent['num_layers'] > 0:
        net = add_recurrent_layers(net, mask_in, **recurrent)

    # now add the "musical model"
    if init_softmax:
        # initialise with the parameters of the input-processor softmax
        crf_params = dict(
            pi=lnn.init.Constant(0),
            tau=lnn.init.Constant(0),
            A=lnn.init.Constant(0),
            W=input_processor_params[-2],
            c=input_processor_params[-1]
        )
    else:
        crf_params = {}

    net = spg.layers.CrfLayer(incoming=net, mask_input=mask_in,
                              num_states=out_size, name='CRF', **crf_params)

    pi, tau, _, A, _ = net.get_params()
    net.params[pi].discard('trainable')
    net.params[tau].discard('trainable')
    net.params[A].discard('trainable')

    # create train function - this one uses the log-likelihood objective
    l2_penalty = lnn.regularization.regularize_network_params(
        net, lnn.regularization.l2) * l2_lambda
    loss = compute_loss(net, target_var, mask_var) + l2_penalty

    # get the network parameters
    params = lnn.layers.get_all_params(net, trainable=True)

    updates = optimiser(loss, params)
    train = theano.function([feature_var, mask_var, target_var], loss,
                            updates=updates)

    # create test and process function. process just computes the prediction
    # without computing the loss, and thus does not need target labels
    test_loss = compute_loss(net, target_var, mask_var) + l2_penalty
    viterbi_out = lnn.layers.get_output(net, mode='viterbi',
                                        deterministic=True)
    test = theano.function([feature_var, mask_var, target_var],
                           [test_loss, viterbi_out])
    process = theano.function([feature_var, mask_var], viterbi_out)

    # return both the feature extraction network as well as the
    # whole thing
    return nn.NeuralNetwork(net, train, test, process)


@ex.config
def config():
    observations = 'results'

    plot = False

    datasource = dict(
        context_size=3
    )

    feature_extractor = None

    input_processor = None

    net = dict(
        l2_lambda=1e-4,
        dense=dict(num_layers=0),
        recurrent=dict(num_layers=0),
        initialise='softmax'  # or 'random' or 'softmax'
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
        batch_size=32,
        max_seq_len=1024,  # at 10 fps, this corresponds to 102 seconds
        early_stop_acc=True,
    )


@ex.named_config
def dense_ip():
    input_processor = dict(
        type='dense',
        freeze_after_train=True,
        fine_tune=False,
        net=dict(
            num_layers=3,
            num_units=256,
            dropout=0.5,
            nonlinearity='rectify',
            batch_norm=False,
            l2_lambda=1e-4,
        ),
        training=dict(
            num_epochs=500,
            early_stop=20,
            batch_size=512,
            early_stop_acc=True,
        ),
        optimiser=dict(
            name='adam',
            params=dict(
                learning_rate=0.0001
            )
        )
    )


@ex.named_config
def recurrent_ip():
    input_processor = dict(
        type='recurrent',
        net=dict(
            l2_lambda=1e-4,
            num_rec_units=128,
            num_layers=3,
            dropout=0.3,
            grad_clip=1.,
            bidirectional=True,
            nonlinearity='rectify'
        ),
        optimiser=dict(
            name='adam',
            params=dict(
                learning_rate=0.001
            )
        ),
        training=dict(
            num_epochs=1000,
            early_stop=20,
            early_stop_acc=True,
            batch_size=64,
            max_seq_len=1024
        )
    )


@ex.automain
def main(_config, _run, observations, datasource, net, feature_extractor,
         input_processor, target, optimiser, training, plot):

    if feature_extractor is None:
        print(Colors.red('ERROR: Specify a feature extractor!'))
        return 1

    if target is None:
        print(Colors.red('ERROR: Specify a target!'))
        return 1

    # Load data sets
    print(Colors.red('Loading data...\n'))

    target_computer = targets.create_target(
        feature_extractor['params']['fps'],
        target
    )
    train_set, val_set, test_set, gt_files = data.create_datasources(
        dataset_names=datasource['datasets'],
        preprocessors=datasource['preprocessors'],
        compute_features=features.create_extractor(feature_extractor),
        compute_targets=target_computer,
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

    # ~~~~~~~~~~~~~~~~~~~~ Train input processors ~~~~~~~~~~~~~~~~~~~~

    if input_processor is not None:
        print(Colors.red('Building input processor network...\n'))
        if input_processor['type'] == 'dense':
            ip_net_cfg = input_processor['net']
            ip_net = build_dense_ip(
                train_set.feature_shape,
                net=ip_net_cfg,
                optimiser=create_optimiser(input_processor['optimiser']),
                out_size=train_set.target_shape[0],
            )

            print(Colors.blue('Input Processor Network:'))
            print(ip_net)
            print('')

            ip_training = input_processor['training']

            print(Colors.red('Starting input processor training...\n'))

            best_ip_params, _, _ = nn.train(
                ip_net, train_set, n_epochs=ip_training['num_epochs'],
                batch_size=ip_training['batch_size'], validation_set=val_set,
                early_stop=ip_training['early_stop'],
                threaded=10,
            )

            print(Colors.red('\nStarting input processor testing...\n'))
            ip_net.set_parameters(best_ip_params)

            with TempDir() as ip_dest_dir:
                pred_files = test.compute_labeling(
                    ip_net, target_computer, test_set, dest_dir=ip_dest_dir,
                    rnn=False
                )

                test_gt_files = dmgr.files.match_files(
                    pred_files, gt_files, test.PREDICTION_EXT, data.GT_EXT
                )

                print(Colors.blue('Input Processor Results:\n'))
                scores = test.compute_average_scores(test_gt_files, pred_files)

                # convert to float so yaml output looks nice
                for k in scores:
                    scores[k] = float(scores[k])
                test.print_scores(scores)

                print('')

                for pf in pred_files:
                    ip_pf = os.path.join(ip_dest_dir, 'ip_' +
                                         os.path.basename(pf))
                    os.rename(pf, ip_pf)
                    ex.add_artifact(ip_pf)

            # make function that recreates the input processor to be
            # included in the complete network later
            def create_input_processor(inp, true_batch_size, true_seq_len):
                resh = lnn.layers.ReshapeLayer(
                    inp, (-1,) + train_set.feature_shape,
                    name='reshape to single')

                layers = dnn.stack_layers(
                    inp=resh,
                    nonlinearity=ip_net_cfg['nonlinearity'],
                    num_layers=ip_net_cfg['num_layers'],
                    num_units=ip_net_cfg['num_units'],
                    dropout=ip_net_cfg['dropout'],
                    batch_norm=ip_net_cfg['batch_norm']
                )

                resh_back = lnn.layers.ReshapeLayer(
                    layers,
                    (true_batch_size, true_seq_len, ip_net_cfg['num_units']),
                    name='reshape back'
                )

                if input_processor['freeze_after_train']:
                    l = resh_back
                    while not isinstance(l, lnn.layers.InputLayer):
                        for p in l.params:
                            l.params[p].discard('trainable')
                            l.params[p].add('frozen')
                        l = l.input_layer

                return resh_back

        else:
            # TODO: RNN input processor!!
            create_input_processor = None
            best_ip_params = None
    else:
        create_input_processor = None
        best_ip_params = None

    best_crf_params = None
    if input_processor is not None and input_processor['freeze_after_train'] and input_processor['fine_tune']:
        # First train the CRF seperately, then train jointly!

        print(Colors.red('Building CRF pre-training network...\n'))

        train_crf_net = build_net(
            feature_shape=train_set.feature_shape,
            batch_size=training['batch_size'],
            max_seq_len=training['max_seq_len'],
            l2_lambda=net['l2_lambda'],
            dense=net['dense'],
            recurrent=net['recurrent'],
            init_softmax=net['init_softmax'],
            optimiser=create_optimiser(optimiser),
            out_size=train_set.target_shape[0],
            input_processor=create_input_processor,
            input_processor_params=best_ip_params
        )

        print(Colors.blue('CRF pre-training Neural Network:'))
        print(train_crf_net)
        print('')

        print(Colors.red('Starting CRF pre-training...\n'))

        with TempDir() as crf_dest_dir:

            best_crf_params, _, _ = nn.train(
                train_crf_net, train_set, n_epochs=training['num_epochs'],
                batch_size=training['batch_size'], validation_set=val_set,
                early_stop=training['early_stop'],
                early_stop_acc=training['early_stop_acc'],
                batch_iterator=dmgr.iterators.iterate_datasources,
                sequence_length=training['max_seq_len'],
                threaded=10
            )

            print(Colors.red('\nStarting pre-trained CRF testing...\n'))

            del train_crf_net

            # build test crf with batch size 1 and no max sequence length
            test_crf_net = build_net(
                feature_shape=test_set.feature_shape,
                batch_size=1,
                max_seq_len=None,
                l2_lambda=net['l2_lambda'],
                dense=net['dense'],
                recurrent=net['recurrent'],
                init_softmax=net['init_softmax'],
                optimiser=create_optimiser(optimiser),
                out_size=test_set.target_shape[0],
                input_processor=create_input_processor,
                input_processor_params=best_ip_params
            )

            # load previously learnt parameters
            test_crf_net.set_parameters(best_crf_params)

            param_file = os.path.join(crf_dest_dir, 'crf_pretrain_params.pkl')
            test_crf_net.save_parameters(param_file)
            ex.add_artifact(param_file)

            pred_files = test.compute_labeling(
                test_crf_net, target_computer, test_set, dest_dir=crf_dest_dir,
                rnn=True
            )

            test_gt_files = dmgr.files.match_files(
                pred_files, gt_files, test.PREDICTION_EXT, data.GT_EXT
            )

            print(Colors.blue('Pre-Trained CRF Results:\n'))

            scores = test.compute_average_scores(test_gt_files, pred_files)
            # convert to float so yaml output looks nice
            for k in scores:
                scores[k] = float(scores[k])
            test.print_scores(scores)

            print('')

            for pf in pred_files:
                crf_pf = os.path.join(crf_dest_dir, 'crf_' +
                                      os.path.basename(pf))
                os.rename(pf, crf_pf)
                ex.add_artifact(crf_pf)

            # turn off parameter freeze
            input_processor['freeze_after_train'] = False
            optimiser['params']['learning_rate'] = 0.000001

    # ~~~~~~~~~~~~~~~~~~~~ Train Network ~~~~~~~~~~~~~~~~~~~~

    # build network
    print(Colors.red('Building network...\n'))

    train_crf_net = build_net(
        feature_shape=train_set.feature_shape,
        batch_size=training['batch_size'],
        max_seq_len=training['max_seq_len'],
        l2_lambda=net['l2_lambda'],
        dense=net['dense'],
        recurrent=net['recurrent'],
        init_softmax=net['init_softmax'],
        optimiser=create_optimiser(optimiser),
        out_size=train_set.target_shape[0],
        input_processor=create_input_processor,
        input_processor_params=best_ip_params
    )

    if best_crf_params is not None:
        train_crf_net.set_parameters(best_crf_params)

    print(Colors.blue('Neural Network:'))
    print(train_crf_net)
    print('')

    print(Colors.red('Starting training...\n'))

    with TempDir() as crf_dest_dir:

        # updates = [ParamSaver(ex, train_neural_net, dest_dir)]

        if plot:
            plot_file = os.path.join(crf_dest_dir, 'plot.pdf')
            updates = [CrfPlotter(train_crf_net.network, plot_file)]
        else:
            updates = []

        best_params, train_losses, val_losses = nn.train(
            train_crf_net, train_set, n_epochs=training['num_epochs'],
            batch_size=training['batch_size'], validation_set=val_set,
            early_stop=training['early_stop'],
            early_stop_acc=training['early_stop_acc'],
            batch_iterator=dmgr.iterators.iterate_datasources,
            sequence_length=training['max_seq_len'],
            updates=updates,
            threaded=10
        )

        if plot:
            updates[-1].close()
            ex.add_artifact(plot_file)

        print(Colors.red('\nStarting testing...'))

        del train_crf_net

        # build test crf with batch size 1 and no max sequence length
        test_crf_net = build_net(
            feature_shape=test_set.feature_shape,
            batch_size=1,
            max_seq_len=None,
            l2_lambda=net['l2_lambda'],
            dense=net['dense'],
            recurrent=net['recurrent'],
            optimiser=create_optimiser(optimiser),
            init_softmax=net['init_softmax'],
            out_size=test_set.target_shape[0],
            input_processor=create_input_processor,
            input_processor_params=best_ip_params
        )

        # load previously learnt parameters
        test_crf_net.set_parameters(best_params)

        param_file = os.path.join(crf_dest_dir, 'params.pkl')
        test_crf_net.save_parameters(param_file)
        ex.add_artifact(param_file)

        pred_files = test.compute_labeling(
            test_crf_net, target_computer, test_set, dest_dir=crf_dest_dir,
            rnn=True
        )

        test_gt_files = dmgr.files.match_files(
            pred_files, gt_files, test.PREDICTION_EXT, data.GT_EXT
        )

        print(Colors.blue('\nResults:\n'))

        scores = test.compute_average_scores(test_gt_files, pred_files)
        # convert to float so yaml output looks nice
        for k in scores:
            scores[k] = float(scores[k])
        test.print_scores(scores)

        result_file = os.path.join(crf_dest_dir, 'results.yaml')
        yaml.dump(dict(scores=scores,
                       train_losses=map(float, train_losses),
                       val_losses=map(float, val_losses)),
                  open(result_file, 'w'))
        ex.add_artifact(result_file)

        for pf in pred_files:
            ex.add_artifact(pf)

    print('')
