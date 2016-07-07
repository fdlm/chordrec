import numpy as np
import pickle
from docopt import docopt

USAGE = """
create_crf_init_params.py - creates initial crf parameters from a learned
                            gap convnet.

Usage:
    create_crf_init_params.py <convnet_params> <crf_params>

Arguments:
    <convnet_params>  pickle file containing the learned convnet parameters
    <crf_params>  file where the initial crf parameters should be stored
"""

args = docopt(USAGE)

params = pickle.load(open(args['<convnet_params>']))
conv, beta, gamma, mean, inv_std = params[-5:]

c = (beta - mean * gamma * inv_std)
W = (conv.reshape(conv.shape[:2]) * gamma[:, np.newaxis] *
     inv_std[:, np.newaxis]).T
pi = np.zeros_like(c)
tau = np.zeros_like(c)
A = np.zeros((len(beta), len(beta)))

pickle.dump([pi.astype(np.float32),
             tau.astype(np.float32),
             c.astype(np.float32),
             A.astype(np.float32),
             W.astype(np.float32)], open(args['<crf_params>'], 'w'))
