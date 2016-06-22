import madmom as mm
import pickle
from glob import glob
from docopt import docopt
from os.path import join

USAGE = """
create_madmom_deep_chroma_model.py - creates madmom models for the
                                     DeepChromaProcessor.

Usage:
    create_madmom_deep_chroma_model.py <param_dir> [<dst_name>]

Arguments:
    <param_dir>  directory containing the parameter files (params_fold_x.pkl)
    <dst_name>  name format for destination files. '{}' will be replaced
                with the model number [default: chroma_nn_{}.pkl]
"""

args = docopt(USAGE)

args['<dst_name>'] = args['<dst_name>'] or 'chroma_dnn_{}.pkl'

param_files = glob(join(args['<param_dir>'], 'params*.pkl'))

for nid, f in enumerate(param_files):
    p = pickle.load(open(f))
    nn = mm.ml.nn.NeuralNetwork([
        mm.ml.nn.layers.FeedForwardLayer(
            p[i], p[i+1],
            # relu layers, but last layer is sigmoid
            mm.ml.nn.activations.relu if i < len(p) - 2 else
            mm.ml.nn.activations.sigmoid
        )
        for i in range(0, len(p) - 2, 2)
    ])
    nn.dump(args['<dst_name>'].format(nid + 1))

