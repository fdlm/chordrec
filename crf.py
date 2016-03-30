from __future__ import print_function
import os
import collections
import theano
import theano.tensor as tt
import lasagne as lnn
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
import convnet
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


def build_net(feature_shape, out_size, input_processor, crf, fine_tuning,
              ip_optimiser, crf_optimiser, fine_tune_optimiser):

    # input variables
    feature_var = (tt.tensor4('feature_input', dtype='float32')
                   if len(feature_shape) > 1 else
                   tt.tensor3('feature_input', dtype='float32'))

    target_var = tt.tensor3('target_output', dtype='float32')
    mask_var = tt.matrix('mask_input', dtype='float32')

    # stack more layers (in this case create a CRF)
    net = lnn.layers.InputLayer(
        name='input', shape=(None, None) + feature_shape,
        input_var=feature_var
    )

    mask_in = lnn.layers.InputLayer(
        name='mask', input_var=mask_var, shape=(None, None)
    )

    true_batch_size, true_seq_len = feature_var.shape[:2]

    net_has_ip = False
    if input_processor and input_processor['type'] in ['dense', 'conv']:
        net_has_ip = True
        ip_net_cfg = input_processor['net']

        if input_processor['type'] == 'dense':
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
        else:  # conv
            net = lnn.layers.ReshapeLayer(net, (-1, 1) + feature_shape,
                                          name='reshape to single')
            # first, stack convolutive layers
            net = convnet.stack_layers(
                net=net,
                batch_norm=ip_net_cfg['batch_norm'],
                convs=[ip_net_cfg[c] for c in ['conv1', 'conv2', 'conv3']]
            )
            # then, add dense layers
            net = dnn.stack_layers(net, **ip_net_cfg['dense'])
            net = lnn.layers.ReshapeLayer(
                net,
                (true_batch_size, true_seq_len,
                 ip_net_cfg['dense']['num_units']),
                name='reshape back'
            )

        # tag layers as input_processor
        for l in lnn.layers.get_all_layers(net):
            if not isinstance(l, lnn.layers.InputLayer):
                for p in l.params:
                    l.params[p].add('input_processor')

    elif input_processor is not None:
        raise RuntimeError('Input processor {} does not exist'.format(
            input_processor['type']
        ))

    if net_has_ip and input_processor['pre_train']:
        ip_output_layer = lnn.layers.DenseLayer(
            net.input_layer, name='ip_output', num_units=out_size,
            nonlinearity=lnn.nonlinearities.softmax
        )
    else:
        ip_output_layer = None

    # now add the "musical model"
    if crf['net']['initialise'] == 'softmax':
        # initialise with the parameters of the crf such that there is no
        # interaction with consecutive predictions, making it behave like a
        # softmax (this will change during training). if we have initial
        # softmax parameters from the input processor, use these.
        crf_params = dict(
            pi=lnn.init.Constant(0),
            tau=lnn.init.Constant(0),
            A=lnn.init.Constant(0),
        )
        if ip_output_layer is not None:
            crf_params['W'] = ip_output_layer.W
            crf_params['c'] = ip_output_layer.b
    else:
        crf_params = {}

    net = spg.layers.CrfLayer(incoming=net, mask_input=mask_in,
                              num_states=out_size, name='CRF', **crf_params)

    for p in net.params:
        net.params[p].add('crf')

    # create input processor training network, if necessary
    if net_has_ip and input_processor['pre_train']:
        ip_pred = lnn.layers.get_output(ip_output_layer)
        ip_l2_penalty = (lnn.regularization.regularize_network_params(
            ip_output_layer, lnn.regularization.l2) *
            input_processor['net']['l2'])
        # reshape target var from (batch_size, seq_len, ...)
        # to (batch_size, ...). assume that seq_len is 1!!
        ip_loss = dnn.compute_loss(ip_pred, target_var[:, 0, :]) +\
            ip_l2_penalty
        # the input_processor tag should not be necessary, but to be sure...
        ip_params = lnn.layers.get_all_params(ip_output_layer, trainable=True,
                                              input_processor=True)
        ip_updates = ip_optimiser(ip_loss, ip_params)
        ip_train = theano.function([feature_var, target_var], ip_loss,
                                   updates=ip_updates)
        ip_test_pred = lnn.layers.get_output(ip_output_layer,
                                             deterministic=True)
        ip_test_loss = dnn.compute_loss(ip_pred, target_var[:, 0, :]) +\
            ip_l2_penalty
        ip_test = theano.function([feature_var, target_var],
                                  [ip_test_loss, ip_test_pred])
        ip_process = theano.function([feature_var], ip_test_pred)

        ip_net = nn.NeuralNetwork(ip_output_layer,
                                  ip_train, ip_test, ip_process)
    else:
        ip_net = None

    # create crf train network
    l2_penalty = lnn.regularization.regularize_network_params(
        net, lnn.regularization.l2) * crf['net']['l2']
    loss = compute_loss(net, target_var, mask_var) + l2_penalty
    test_loss = compute_loss(net, target_var, mask_var) + l2_penalty
    viterbi_out = lnn.layers.get_output(net, mode='viterbi',
                                        deterministic=True)
    test = theano.function([feature_var, mask_var, target_var],
                           [test_loss, viterbi_out])
    process = theano.function([feature_var, mask_var], viterbi_out)

    if fine_tuning:
        # get the network parameters
        params = lnn.layers.get_all_params(net, trainable=True)
        updates = fine_tune_optimiser(loss, params)
        train = theano.function([feature_var, mask_var, target_var], loss,
                                updates=updates)
        crf_net = nn.NeuralNetwork(net, train, test, process)
    else:
        crf_net = None

    if crf['pre_train']:
        # get only CRF parameters
        crf_params = lnn.layers.get_all_params(net, trainable=True, crf=True)
        crf_updates = crf_optimiser(loss, crf_params)
        crf_pretrain = theano.function([feature_var, mask_var, target_var],
                                       loss, updates=crf_updates)
        # test and process functions are the same - we are just training
        # a part of the net
        crf_pretrain_net = nn.NeuralNetwork(net, crf_pretrain, test, process)
    else:
        crf_pretrain_net = None

    # return the whole thing, as well as the other nets (None if not existing)
    return crf_net, ip_net, crf_pretrain_net


@ex.config
def config():
    observations = 'results'

    plot = False

    datasource = dict(
        context_size=0
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
            ),
            schedule=None
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
                learning_rate=0.00001
            ),
            schedule=None
        )
    )


@ex.named_config
def dense_ip():
    datasource = dict(
        context_size=7
    )
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
            ),
            schedule=None
        )
    )


@ex.named_config
def conv_ip():
    datasource = dict(
        context_size=7
    )
    input_processor = dict(
        type='conv',
        pre_train=True,
        net=dict(
            batch_norm=True,
            conv1=dict(
                num_layers=2,
                num_filters=8,
                filter_size=(3, 3),
                pool_size=(1, 2),
                dropout=0.5,
            ),
            conv2=dict(
                num_layers=1,
                num_filters=16,
                filter_size=(3, 3),
                pool_size=(1, 2),
                dropout=0.5,
            ),
            conv3={},
            dense=dict(
                num_layers=1,
                num_units=256,
                dropout=0.5,
                nonlinearity='rectify',
                batch_norm=False
            ),
            global_avg_pool=None,
            l2=1e-4,
            l1=0
        ),

        optimiser=dict(
            name='adam',
            params=dict(
                learning_rate=0.001
            ),
            schedule=None
        ),

        training=dict(
            num_epochs=500,
            early_stop=20,
            early_stop_acc=True,
            batch_size=2048,
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


def train_and_test(net, train_set, val_set, test_set, gt_files,
                   test_fold, training, updates, target_computer,
                   dest_dir, name, rnn, **kwargs):

    print(Colors.red('\nStarting {} training...\n'.format(name)))

    if rnn:
        kwargs['sequence_length'] = training['max_seq_len']

    best_params, train_losses, val_losses = nn.train(
        net, train_set, n_epochs=training['num_epochs'],
        batch_size=training['batch_size'], validation_set=val_set,
        early_stop=training['early_stop'],
        early_stop_acc=training['early_stop_acc'],
        threaded=10,
        updates=updates,
        **kwargs
    )

    print(Colors.red('\nStarting {} testing...\n'.format(name)))

    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)

    param_file = os.path.join(
        dest_dir, 'params_fold_{}.pkl'.format(test_fold))
    net.save_parameters(param_file)

    pred_files = test.compute_labeling(
        net, target_computer, test_set, dest_dir=dest_dir,
        use_mask=rnn, add_time_dim=kwargs.get('add_time_dim', False)
    )

    test_gt_files = dmgr.files.match_files(
        pred_files, gt_files, test.PREDICTION_EXT, data.GT_EXT
    )

    print(Colors.blue('{} Results:\n'.format(name)))
    scores = test.compute_average_scores(test_gt_files, pred_files)
    test.print_scores(scores)

    result_file = os.path.join(
        dest_dir, 'results_fold_{}.yaml'.format(test_fold))
    yaml.dump(dict(scores=scores,
                   train_losses=map(float, train_losses),
                   val_losses=map(float, val_losses)),
              open(result_file, 'w'))

    return best_params, pred_files, test_gt_files, dest_dir


def optimiser_and_updates(optimiser):
    opt, lr = create_optimiser(optimiser)
    if 'schedule' in optimiser and optimiser['schedule'] is not None:
        upd = [nn.LearnRateSchedule(learning_rate=lr, **optimiser['schedule'])]
    else:
        upd = []

    return opt, upd


@ex.automain
def main(_config, _run, observations, datasource, feature_extractor, target,
         input_processor, crf, fine_tuning, testing, plot):

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

    all_pred_files = dict(input_processor=[], crf_pretrain=[], fine_tune=[])
    all_gt_files = dict(input_processor=[], crf_pretrain=[], fine_tune=[])
    all_dirs = set()

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

            if testing['test_on_val']:
                test_set = val_set

            print(Colors.blue('Train Set:'))
            print('\t', train_set)

            print(Colors.blue('Validation Set:'))
            print('\t', val_set)

            print(Colors.blue('Test Set:'))
            print('\t', test_set)
            print('')

            print(Colors.red('Building network...\n'))

            if input_processor and input_processor['pre_train']:
                ip_optimiser, ip_updates = optimiser_and_updates(
                    input_processor['optimiser'])
            else:
                ip_optimiser, ip_updates = None, []

            if crf['pre_train']:
                crf_optimiser, crf_updates = optimiser_and_updates(
                    crf['optimiser'])
            else:
                crf_optimiser, crf_updates = None, []

            if fine_tuning:
                fine_tune_optimiser, fine_tune_updates = optimiser_and_updates(
                    fine_tuning['optimiser'])
            else:
                fine_tune_optimiser, fine_tune_updates = None, []

            crf_net, ip_net, crf_pretrain_net = build_net(
                feature_shape=train_set.feature_shape,
                out_size=train_set.target_shape[0],
                input_processor=input_processor,
                crf=crf, fine_tuning=fine_tuning,
                ip_optimiser=ip_optimiser, crf_optimiser=crf_optimiser,
                fine_tune_optimiser=fine_tune_optimiser
            )

            if ip_net:
                print(Colors.blue('Input Processor Network:'))
                print(ip_net)
                print('')

            if crf_pretrain_net:
                print(Colors.blue('CRF Pre-Train Network:'))
                print(crf_pretrain_net)
                print('')

            if crf_net:
                print(Colors.blue('Whole CRF Network:'))
                print(crf_net)
                print('')

            # ~~~~~~~~~~~~~~~~~~~~ Train input processor ~~~~~~~~~~~~~~~~~~~~
            if ip_net is not None:
                best_ip_params, ip_pred, ip_gt, ip_dir = train_and_test(
                    ip_net, train_set, val_set, test_set, gt_files, test_fold,
                    input_processor['training'], ip_updates,
                    target_computer, os.path.join(exp_dir, 'input_processor'),
                    'Input Processor', rnn=False, add_time_dim=True
                )

                all_pred_files['input_processor'] += ip_pred
                all_gt_files['input_processor'] += ip_gt
                all_dirs.add(ip_dir)

            # ~~~~~~~~~~~~~~~~~~~~~~~ Train CRF ~~~~~~~~~~~~~~~~~~~~~~~
            if crf['pre_train']:
                best_crf_params, crf_pred, crf_gt, crf_dir = train_and_test(
                    crf_pretrain_net, train_set, val_set, test_set, gt_files,
                    test_fold, crf['training'], crf_updates,
                    target_computer, os.path.join(exp_dir, 'crf_pretrain'),
                    'CRF pre-train', rnn=True,
                    batch_iterator=dmgr.iterators.iterate_datasources,
                )

                all_pred_files['crf_pretrain'] += crf_pred
                all_gt_files['crf_pretrain'] += crf_gt
                all_dirs.add(crf_dir)

            # ~~~~~~~~~~~~~~~~~~~~ Fine-Tune Network ~~~~~~~~~~~~~~~~~~~~
            if fine_tuning:
                best_ft_params, ft_pred, ft_gt, ft_dir = train_and_test(
                    crf_net, train_set, val_set, test_set, gt_files,
                    test_fold, fine_tuning['training'], fine_tune_updates,
                    target_computer, os.path.join(exp_dir, 'fine_tuned'),
                    'fine-tuned', rnn=True,
                    batch_iterator=dmgr.iterators.iterate_datasources,
                )

                all_pred_files['fine_tune'] += ft_pred
                all_gt_files['fine_tune'] += ft_gt
                all_dirs.add(ft_dir)

        # if there is something to aggregate
        if len(datasource['test_fold']) > 1:
            print(Colors.yellow('\nAggregated Results:\n'))
            for k in all_pred_files:
                if len(all_pred_files[k]) == 0:
                    continue
                scores = test.compute_average_scores(
                    all_gt_files[k], all_pred_files[k])
                print(Colors.blue('\n' + k + '\n'))
                test.print_scores(scores)
                result_file = os.path.join(
                    exp_dir, 'results_{}.yaml'.format(k))
                yaml.dump(dict(scores=scores), open(result_file, 'w'))
                ex.add_artifact(result_file)

        # save all the sub-directories
        for d in all_dirs:
            ex.add_artifact(d)

    print(Colors.magenta('Stopping experiment ' + ex.observers[0].hash()))
