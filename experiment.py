from __future__ import print_function
import os
import yaml
import pickle
import shutil
import hashlib
import tempfile
from functools import partial
from sacred import Experiment
from sacred.observers import RunObserver
import lasagne as lnn
import theano
import numpy as np

import nn
import dmgr
from nn.utils import Colors

import data
import features
import targets
import test


class TempDir:
    """
    Creates a temporary directory to save stuff to
    """
    def __enter__(self):
        self._tmp_dir_path = tempfile.mkdtemp()
        return self._tmp_dir_path

    def __exit__(self, type, value, traceback):
        shutil.rmtree(self._tmp_dir_path)


def create_optimiser(optimiser):
    """
    Creates a function that returns an optimiser and (optional) a learn
    rate schedule
    """

    if optimiser['schedule'] is not None:
        # if we have a learn rate schedule, create a theano shared
        # variable and a corresponding update
        lr = theano.shared(np.float32(optimiser['params']['learning_rate']))

        # create a copy of the optimiser config dict so we do not change
        # it
        from copy import deepcopy
        optimiser = deepcopy(optimiser)
        optimiser['params']['learning_rate'] = lr
        lrs = nn.LearnRateSchedule(learning_rate=lr, **optimiser['schedule'])
    else:
        lrs = None

    return partial(getattr(lnn.updates, optimiser['name']),
                   **optimiser['params']), lrs


def rhash(d):
    """
    Coputes the recursive hash of a dictionary
    :param d:  dictionary to hash
    :return:   hash of dictionary
    """
    m = hashlib.sha1()

    if isinstance(d, dict):
        for _, value in sorted(d.items(), key=lambda (k, v): k):
            m.update(rhash(value))
    else:
        m.update(str(d))

    return m.hexdigest()


def fhash(filename):
    """
    Computes the hash of a file
    :param filename: file to hash
    :return:         hash value of file
    """
    md5 = hashlib.md5()
    with open(filename, 'rb') as f:
        # this needs an empty *byte* string b'' as a sentinel value
        for chunk in iter(lambda: f.read(128 * md5.block_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


class PickleAndSymlinkObserver(RunObserver):

    def __init__(self):
        self.config = None
        self.run = None
        self._hash = None

    def started_event(self, ex_info, host_info, start_time, config, comment):
        self.config = config

        # remember the *exact* configuration used for this run
        config_file = os.path.join(self.config_path(), 'config.yaml')
        with open(config_file, 'w') as f:
            f.write(yaml.dump(self.config))

        self.run = {
            'ex_info': ex_info,
            'host_info': host_info,
            'start_time': start_time,
            'comment': comment
        }

    def hash(self):
        if self._hash is None:
            self._hash = rhash(self.config)

        return self._hash

    def config_path(self):
        if self.config is None:
            raise RuntimeError('tried to get a path without a configuration!')

        config_save_path = os.path.join(self.config['observations'],
                                        self.hash())

        if not os.path.exists(config_save_path):
            os.mkdir(config_save_path)
            os.mkdir(os.path.join(config_save_path, 'resources'))
            os.mkdir(os.path.join(config_save_path, 'artifacts'))
        return config_save_path

    def heartbeat_event(self, info, captured_out, beat_time):
        self.run['info'] = info
        self.run['captured_out'] = captured_out
        self.run['beat_time'] = beat_time

    def completed_event(self, stop_time, result):
        run_file = os.path.join(self.config_path(), 'completed.pkl')
        with open(run_file, 'w') as f:
            pickle.dump(self.run, f)

    def interrupted_event(self, interrupt_time):
        self.run['interrupt_time'] = interrupt_time
        interrupted_file = os.path.join(self.config_path(), 'interrupted.pkl')
        with open(interrupted_file, 'w') as f:
            pickle.dump(self.run, f)

    def failed_event(self, fail_time, fail_trace):
        self.run['fail_time'] = fail_time
        self.run['fail_trace'] = fail_trace

        fail_file = os.path.join(self.config_path(), 'failed.pkl')
        with open(fail_file, 'w') as f:
            pickle.dump(self.run, f)

        fail_file = os.path.join(self.config_path(), 'failed_trace.txt')
        with open(fail_file, 'w') as f:
            f.write(''.join(fail_trace))

    def resource_event(self, filename):
        """
        link a used file (this is where we could have distributed storage)...
        """
        linkname = os.path.join(self.config_path(), 'resources',
                                fhash(filename))
        if not os.path.exists(linkname):
            os.symlink(filename, linkname)

    def artifact_event(self, filename):
        """
        move an artifact from a temporary space to the actual observations
        directory for this run
        """
        newname = os.path.join(self.config_path(), 'artifacts',
                               os.path.basename(filename))
        shutil.move(filename, newname)

    def get_artifact_path(self, path):
        return os.path.join(self.config_path(), 'artifacts', path)


class ParamSaver:

    def __init__(self, ex, net, tmp_dir):
        self.ex = ex
        self.tmp_dir = tmp_dir
        self.net = net

    def __call__(self, epoch):
        fn = os.path.join(self.tmp_dir, 'params_{}.pkl'.format(epoch))
        self.net.save_parameters(fn)
        self.ex.add_artifact(fn)


def setup(name):
    ex = Experiment(name)
    ex.observers.append(PickleAndSymlinkObserver())
    data.add_sacred_config(ex)
    features.add_sacred_config(ex)
    targets.add_sacred_config(ex)
    return ex


def run(ex, build_fn, loss_fn,
        datasource, net, feature_extractor, regularisation, target,
        optimiser, training, testing):

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

    if not isinstance(datasource['test_fold'], list):
        datasource['test_fold'] = [datasource['test_fold']]

    if not isinstance(datasource['val_fold'], list):
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

            if testing['test_on_val']:
                test_set = val_set

            print(Colors.blue('Train Set:'))
            print('\t', train_set)
            print(Colors.blue('Validation Set:'))
            print('\t', val_set)
            print(Colors.blue('Test Set:'))
            print('\t', test_set)
            print('')

            # build network
            print(Colors.red('Building network...\n'))

            nnet_vars = build_fn(
                feature_shape=train_set.dshape,
                out_size=train_set.tshape[0],
                net=net
            )

            if len(nnet_vars) == 3:
                neural_net, input_var, target_var = nnet_vars
                mask_var = None

                it = training.get('iterator', 'BatchIterator')

                if it == 'BatchIterator':
                    train_batches = dmgr.iterators.BatchIterator(
                        train_set, training['batch_size'], shuffle=True,
                        expand=True
                    )
                elif it == 'ClassBalancedIterator':
                    train_batches = dmgr.iterators.ClassBalancedIterator(
                        train_set, training['batch_size']
                    )
                else:
                    raise ValueError('Unknown Batch Iterator: {}'.format(it))

                validation_batches = dmgr.iterators.BatchIterator(
                    val_set, training['batch_size'], shuffle=False,
                    expand=False
                )

                use_mask = False
            elif len(nnet_vars) == 4:
                neural_net, input_var, mask_var, target_var = nnet_vars

                train_batches = dmgr.iterators.DatasourceIterator(
                    train_set, training['batch_size'], shuffle=True,
                    expand=True, max_seq_len=training['max_seq_len']
                )

                validation_batches = dmgr.iterators.DatasourceIterator(
                    val_set, training['batch_size'], shuffle=False,
                    expand=False
                )
                use_mask = True
            else:
                raise ValueError('Invalid number of return values in build_fn')

            opt, lrs = create_optimiser(optimiser)

            train_fn = nn.compile_train_fn(
                neural_net, input_var, target_var,
                loss_fn=loss_fn, opt_fn=opt, mask_var=mask_var,
                **regularisation
            )

            test_fn = nn.compile_test_func(
                neural_net, input_var, target_var,
                loss_fn=loss_fn, mask_var=mask_var,
                **regularisation
            )

            process_fn = nn.compile_process_func(
                neural_net, input_var, mask_var=mask_var)

            print(Colors.blue('Neural Network:'))
            print(nn.to_string(neural_net))
            print('')

            print(Colors.red('Starting training...\n'))

            train_losses, val_losses, _, val_accs = nn.train(
                network=neural_net,
                train_fn=train_fn, train_batches=train_batches,
                test_fn=test_fn, validation_batches=validation_batches,
                threads=10, callbacks=[lrs] if lrs else [],
                num_epochs=training['num_epochs'],
                early_stop=training['early_stop'],
                early_stop_acc=training['early_stop_acc']
            )

            print(Colors.red('\nStarting testing...\n'))

            param_file = os.path.join(
                exp_dir, 'params_fold_{}.pkl'.format(test_fold))
            nn.save_params(neural_net, param_file)
            ex.add_artifact(param_file)

            pred_files = test.compute_labeling(
                process_fn, target_computer, test_set, dest_dir=exp_dir,
                use_mask=use_mask
            )

            test_gt_files = dmgr.files.match_files(
                pred_files, test.PREDICTION_EXT, gt_files, data.GT_EXT
            )

            all_pred_files += pred_files
            all_gt_files += test_gt_files

            print(Colors.blue('Results:'))
            scores = test.compute_average_scores(test_gt_files, pred_files)
            test.print_scores(scores)
            result_file = os.path.join(
                exp_dir, 'results_fold_{}.yaml'.format(test_fold))
            yaml.dump(dict(scores=scores,
                           train_losses=map(float, train_losses),
                           val_losses=map(float, val_losses),
                           val_accs=map(float, val_accs)),
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
