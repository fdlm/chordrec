from __future__ import print_function
import os
import collections
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


def build_net(feature_shape, batch_size, max_seq_len, out_size, optimiser,
              input_processor, input_processor_params, train_ip_params,
              l2, init_softmax):

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

    if input_processor['type'] == 'dense':
        ip_net_cfg = input_processor['net']
        net = lnn.layers.ReshapeLayer(net, (-1,) + feature_shape,
                                      name='reshape to single')
        net = dnn.stack_layers(
            inp=net,
            batch_norm=ip_net_cfg['batch_norm'],
            nonlinearity=ip_net_cfg['nonlinearity'],
            num_layers=ip_net_cfg['num_layers'],
            num_units=ip_net_cfg['num_units'],
            dropout=ip_net_cfg['dropout']
        )
        net = lnn.layers.ReshapeLayer(
            net,
            (true_batch_size, true_seq_len, ip_net_cfg['num_units']),
            name='reshape back'
        )

        lnn.layers.set_all_param_values(net, input_processor_params[:-2])

        if not train_ip_params:
            l = net
            while not isinstance(l, lnn.layers.InputLayer):
                for p in l.params:
                    l.params[p].discard('trainable')
                l = l.input_layer

    elif input_processor['type'] is not None:
        raise RuntimeError('Input processor {} does not exist'.format(
            input_processor['type']
        ))

    # now add the "musical model"
    if init_softmax:
        # initialise with the parameters of the crf such that there is no
        # interaction with consecutive predictions, making it behave like a
        # softmax (this will change during training). if we have initial
        # softmax parameters from the input processor, use these.
        crf_params = dict(
            pi=lnn.init.Constant(0),
            tau=lnn.init.Constant(0),
            A=lnn.init.Constant(0),
        )
        if input_processor_params is not None:
            crf_params['W'] = input_processor_params[-2],
            crf_params['c'] = input_processor_params[-1]
    else:
        crf_params = {}

    net = spg.layers.CrfLayer(incoming=net, mask_input=mask_in,
                              num_states=out_size, name='CRF', **crf_params)

    # create train function - this one uses the log-likelihood objective
    l2_penalty = lnn.regularization.regularize_network_params(
        net, lnn.regularization.l2) * l2
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

    crf = dict(
        pre_train=True,

        net=dict(
            l2=1e-4,
            initialise='softmax'  # or 'random' or 'softmax'
        ),

        optimiser=dict(
            name='adam',
            params=dict(
                learning_rate=0.002
            )
        ),

        training=dict(
            num_epochs=1000,
            early_stop=20,
            batch_size=32,
            max_seq_len=1024,  # at 10 fps, this corresponds to 102 seconds
            early_stop_acc=True,
        )
    )

    fine_tuning = False


@ex.named_config
def fine_tune():
    fine_tuning = dict(
        training=dict(
            num_epochs=1000,
            early_stop=20,
            batch_size=32,
            max_seq_len=1024,
            early_stop_acc=True
        ),
        optimiser=dict(
            name='adam',
            params=dict(
                learning_rate=0.0001
            )
        )
    )


@ex.named_config
def dense_ip():
    input_processor = dict(
        type='dense',
        pre_train=True,
        net=dict(
            num_layers=3,
            num_units=256,
            dropout=0.5,
            nonlinearity='rectify',
            batch_norm=False,
            l2=1e-4,
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


# @ex.named_config
# def recurrent_ip():
#     input_processor = dict(
#         type='recurrent',
#         net=dict(
#             l2_lambda=1e-4,
#             num_rec_units=128,
#             num_layers=3,
#             dropout=0.3,
#             grad_clip=1.,
#             bidirectional=True,
#             nonlinearity='rectify'
#         ),
#         optimiser=dict(
#             name='adam',
#             params=dict(
#                 learning_rate=0.001
#             )
#         ),
#         training=dict(
#             num_epochs=1000,
#             early_stop=20,
#             early_stop_acc=True,
#             batch_size=64,
#             max_seq_len=1024
#         )
#     )


def pretrain_input_processor(train_set, val_set, test_set, gt_files, test_fold,
                             input_processor, target_computer, exp_dir):

    print(Colors.red('Building input processor network...\n'))

    net = input_processor['net']
    optimiser = input_processor['optimiser']
    training = input_processor['training']

    opt, learn_rate = create_optimiser(optimiser)

    if input_processor['type'] == 'dense':

        ip_net = dnn.build_net(
            feature_shape=train_set.feature_shape,
            optimiser=opt,
            out_size=train_set.target_shape[0],
            **net
        )

        print(Colors.blue('Input Processor Network:'))
        print(ip_net)
        print('')

        print(Colors.red('Starting input processor training...\n'))

        updates = []
        if optimiser['schedule'] is not None:
            updates.append(
                nn.LearnRateSchedule(
                    learn_rate=learn_rate, **optimiser['schedule'])
            )

        best_ip_params, ip_train_losses, ip_val_losses = nn.train(
            ip_net, train_set, n_epochs=training['num_epochs'],
            batch_size=training['batch_size'], validation_set=val_set,
            early_stop=training['early_stop'],
            early_stop_acc=training['early_stop_acc'],
            threaded=10,
            updates=updates
        )

        print(Colors.red('\nStarting input processor testing...\n'))

        ip_dest_dir = os.path.join(exp_dir, 'input_processor')
        if not os.path.exists(ip_dest_dir):
            os.mkdir(ip_dest_dir)

        param_file = os.path.join(
            ip_dest_dir, 'params_fold_{}.pkl'.format(test_fold))
        ip_net.save_parameters(param_file)

        pred_files = test.compute_labeling(
            ip_net, target_computer, test_set, dest_dir=ip_dest_dir,
            rnn=False
        )

        test_gt_files = dmgr.files.match_files(
            pred_files, gt_files, test.PREDICTION_EXT, data.GT_EXT
        )

        print(Colors.blue('Input Processor Results:\n'))
        scores = test.compute_average_scores(test_gt_files, pred_files)
        test.print_scores(scores)

        result_file = os.path.join(
            exp_dir, 'results_fold_{}.yaml'.format(test_fold))
        yaml.dump(dict(scores=scores,
                       train_losses=map(float, ip_train_losses),
                       val_losses=map(float, ip_val_losses)),
                  open(result_file, 'w'))

        # add the whole directory as artifact to the experiment. this
        # probably does not work with other observers
        ex.add_artifact(ip_dest_dir)

        return best_ip_params

    else:
        raise NotImplementedError('other input processors not implemented yet')


def pretrain_crf(train_set, val_set, test_set, gt_files, test_fold,
                 crf, input_processor, input_processor_params, target_computer,
                 exp_dir):

    print(Colors.red('Building CRF pre-training network...\n'))

    net = crf['net']
    optimiser = crf['optimiser']
    training = crf['training']

    opt, learn_rate = create_optimiser(optimiser)

    train_crf_net = build_net(
        feature_shape=train_set.feature_shape,
        batch_size=training['batch_size'],
        max_seq_len=training['max_seq_len'],
        out_size=train_set.target_shape[0],
        optimiser=opt,
        input_processor=input_processor,
        input_processor_params=input_processor_params,
        train_ip_params=False,
        **net
    )

    print(Colors.blue('CRF pre-training Neural Network:'))
    print(train_crf_net)
    print('')

    print(Colors.red('Starting CRF pre-training...\n'))

    updates = []
    if optimiser['schedule'] is not None:
        updates.append(
            nn.LearnRateSchedule(
                learn_rate=learn_rate, **optimiser['schedule'])
        )

    best_crf_params, crf_train_losses, crf_val_losses = nn.train(
        train_crf_net, train_set, n_epochs=training['num_epochs'],
        batch_size=training['batch_size'], validation_set=val_set,
        early_stop=training['early_stop'],
        early_stop_acc=training['early_stop_acc'],
        batch_iterator=dmgr.iterators.iterate_datasources,
        sequence_length=training['max_seq_len'],
        threaded=10,
        updates=updates
    )

    print(Colors.red('\nStarting pre-trained CRF testing...\n'))

    crf_dest_dir = os.path.join(exp_dir, 'crf')
    if not os.path.exists(crf_dest_dir):
        os.mkdir(crf_dest_dir)

    param_file = os.path.join(
        crf_dest_dir, 'params_fold_{}.pkl'.format(test_fold))
    train_crf_net.save_parameters(param_file)
    del train_crf_net  # we do not need it anymore

    # build test crf with batch size 1 and no max sequence length
    test_crf_net = build_net(
        feature_shape=test_set.feature_shape,
        batch_size=1,
        max_seq_len=None,
        out_size=test_set.target_shape[0],
        optimiser=opt,
        input_processor=input_processor,
        input_processor_params=input_processor_params,
        **crf
    )

    # load previously learnt parameters
    test_crf_net.set_parameters(best_crf_params)

    pred_files = test.compute_labeling(
        test_crf_net, target_computer, test_set, dest_dir=crf_dest_dir,
        rnn=True
    )

    test_gt_files = dmgr.files.match_files(
        pred_files, gt_files, test.PREDICTION_EXT, data.GT_EXT
    )

    print(Colors.blue('\nPre-Trained CRF Results:\n'))
    scores = test.compute_average_scores(test_gt_files, pred_files)
    test.print_scores(scores)

    result_file = os.path.join(
        exp_dir, 'results_fold_{}.yaml'.format(test_fold))
    yaml.dump(dict(scores=scores,
                   train_losses=map(float, crf_train_losses),
                   val_losses=map(float, crf_val_losses)),
              open(result_file, 'w'))

    # add the whole directory as artifact to the experiment. this
    # probably does not work with other observers
    ex.add_artifact(crf_dest_dir)

    return best_crf_params


def fine_tune_network(train_set, val_set, crf, params, input_processor,
                      input_processor_params, fine_tuning):
    # build network
    print(Colors.red('Building fine-tuning network...\n'))

    net = crf['net']
    optimiser = fine_tuning['optimiser']
    training = fine_tuning['training']

    opt, learn_rate = create_optimiser(optimiser)

    train_crf_net = build_net(
        feature_shape=train_set.feature_shape,
        batch_size=training['batch_size'],
        max_seq_len=training['max_seq_len'],
        out_size=train_set.target_shape[0],
        optimiser=opt,
        input_processor=input_processor,
        input_processor_params=input_processor_params,
        train_ip_params=True,
        **net
    )

    if params is not None:
        train_crf_net.set_parameters(params)

    print(Colors.blue('Neural Network:'))
    print(train_crf_net)
    print('')

    print(Colors.red('Starting fine-tuning...\n'))

    updates = []
    if optimiser['schedule'] is not None:
        updates.append(
            nn.LearnRateSchedule(
                learn_rate=learn_rate, **optimiser['schedule'])
        )

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

    return best_params, train_losses, val_losses


@ex.automain
def main(_config, _run, observations, datasource, feature_extractor, target,
         input_processor, crf, fine_tuning, plot):

    if feature_extractor is None:
        print(Colors.red('ERROR: Specify a feature extractor!'))
        return 1

    if target is None:
        print(Colors.red('ERROR: Specify a target!'))
        return 1

    target_computer = targets.create_target(
        feature_extractor['params']['fps'],
        target
    )

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

            train_set, val_set, test_set, gt_files = data.create_datasources(
                dataset_names=datasource['datasets'],
                preprocessors=datasource['preprocessors'],
                compute_features=features.create_extractor(feature_extractor),
                compute_targets=target_computer,
                context_size=datasource['context_size'],
                test_fold=test_fold,
                val_fold=val_fold,
                cached=datasource['cached'],
            )

            print(Colors.blue('Train Set:'))
            print('\t', train_set)

            print(Colors.blue('Validation Set:'))
            print('\t', val_set)

            print(Colors.blue('Test Set:'))
            print('\t', test_set)
            print('')

            # ~~~~~~~~~~~~~~~~~~~~ Train input processor ~~~~~~~~~~~~~~~~~~~~
            if input_processor is not None and input_processor['pre_train']:
                best_ip_params = pretrain_input_processor(
                    train_set, val_set, test_set, gt_files, test_fold,
                    input_processor, target_computer, exp_dir
                )
            else:
                best_ip_params = None

            # ~~~~~~~~~~~~~~~~~~~~~~~ Train CRF ~~~~~~~~~~~~~~~~~~~~~~~
            if crf['pre_train']:
                best_crf_params = pretrain_crf(
                    train_set, val_set, test_set, gt_files, test_fold, crf,
                    input_processor, best_ip_params, target_computer, exp_dir
                )
            else:
                best_crf_params = None

            # ~~~~~~~~~~~~~~~~~~~~ Fine-Tune Network ~~~~~~~~~~~~~~~~~~~~
            if fine_tuning:
                best_crf_params, train_losses, val_losses = fine_tune_network(
                    train_set, val_set, crf, best_crf_params, input_processor,
                    best_ip_params, fine_tuning
                )

            print(Colors.red('\nStarting testing...'))

            # build test crf with batch size 1 and no max sequence length
            test_crf_net = build_net(
                feature_shape=test_set.feature_shape,
                batch_size=1,
                max_seq_len=None,
                out_size=test_set.target_shape[0],
                optimiser=create_optimiser(crf['optimiser'])[0],
                input_processor=input_processor,
                input_processor_params=best_ip_params,
                **crf
            )

            # load previously learnt parameters
            test_crf_net.set_parameters(best_crf_params)

            param_file = os.path.join(
                exp_dir, 'params_fold_{}.pkl'.format(test_fold))
            test_crf_net.save_parameters(param_file)
            ex.add_artifact(param_file)

            pred_files = test.compute_labeling(
                test_crf_net, target_computer, test_set, dest_dir=exp_dir,
                rnn=True
            )

            test_gt_files = dmgr.files.match_files(
                pred_files, gt_files, test.PREDICTION_EXT, data.GT_EXT
            )

            all_pred_files += pred_files
            all_gt_files += test_gt_files

            print(Colors.blue('\nResults:\n'))
            scores = test.compute_average_scores(test_gt_files, pred_files)
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
            test.print_scores(scores)
            result_file = os.path.join(exp_dir, 'results.yaml')
            yaml.dump(dict(scores=scores), open(result_file, 'w'))
            ex.add_artifact(result_file)

        for pf in all_pred_files:
            ex.add_artifact(pf)

    print(Colors.magenta('Stopping experiment ' + ex.observers[0].hash()))
