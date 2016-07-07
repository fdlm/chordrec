import numpy as np
import pickle
import os
from docopt import docopt
from glob import glob
from os.path import join, exists


USAGE = """
create_crf_init_params.py - creates initial crf parameters from a learned
                            gap convnet.

Usage:
    create_crf_init_params.py <src_dir> <dst_dir>

Arguments:
    <src_dir>  directory containing the CNN parameter files for each fold
    <dst_dir>  directory where to store the initial CRF parameters
"""

args = docopt(USAGE)
param_files = glob(join(args['<src_dir>'], 'params*.pkl'))

if not exists(args['<dst_dir>']):
    os.makedirs(args['<dst_dir>'])

for fold, pfile in enumerate(param_files):
    params = pickle.load(open(pfile))
    conv, beta, gamma, mean, inv_std = params[-5:]

    c = (beta - mean * gamma * inv_std)
    W = (conv.reshape(conv.shape[:2]) * gamma[:, np.newaxis] *
         inv_std[:, np.newaxis]).T
    pi = np.zeros_like(c)
    tau = np.zeros_like(c)
    A = np.zeros((len(beta), len(beta)))

    dst_file = join(args['<dst_dir>'], 'crf_init_params_{}.pkl'.format(fold))

    pickle.dump([pi.astype(np.float32),
                 tau.astype(np.float32),
                 c.astype(np.float32),
                 A.astype(np.float32),
                 W.astype(np.float32)], open(dst_file, 'w'))
