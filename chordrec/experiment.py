from __future__ import print_function
import os
import yaml
import pickle
import shutil
import hashlib
import tempfile
import sys
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
import augmenters


class TempDir:
    """
    Creates a temporary directory to save stuff to
    """
    def __enter__(self):
        self._tmp_dir_path = tempfile.mkdtemp()
        return self._tmp_dir_path

    def __exit__(self, type, value, traceback):
        shutil.rmtree(self._tmp_dir_path)


def compute_features(process_fn, agg_dataset, dest_dir, use_mask,
                     batch_size, extension):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    else:
        if not os.path.isdir(dest_dir):
            print(Colors.red('Destination path exists but is not a directory!'),
                  file=sys.stderr)
            return

    iterate_batches = dmgr.iterators.iterate_batches

    feature_files = []

    for ds_idx in range(agg_dataset.n_datasources):
        ds = agg_dataset.datasource(ds_idx)

        feats = []
        for data, _ in iterate_batches(ds, batch_size or ds.n_data,
                                       randomise=False, expand=False):
            if use_mask:
                data = data[np.newaxis, :]
                mask = np.ones(data.shape[:2], dtype=np.float32)

                f = process_fn(data, mask)[0]
            else:
                f = process_fn(data)
            feats.append(f)

        feats = np.concatenate(feats)
        feat_file = os.path.join(dest_dir, ds.name + extension)
        np.save(feat_file, feats)
        feature_files.append(feat_file)

    return feature_files


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
            os.makedirs(os.path.join(config_save_path, 'resources'))
            os.makedirs(os.path.join(config_save_path, 'artifacts'))
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
    augmenters.add_sacred_config(ex)
    return ex


