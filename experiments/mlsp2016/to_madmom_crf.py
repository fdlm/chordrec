import madmom as mm
import pickle
from glob import glob
from docopt import docopt
from os.path import join

USAGE = """
create_madmom_deep_chroma_model.py - creates a madmom crf that predicts chords from deep chroma
vectors.

Usage:
    to_madmom_crf.py <param_dir> [<dst_name>]

Arguments:
    <param_dir>  directory containing the parameter files
    <dst_name>  name format for destination files. '{}' will be replaced
                with the model number [default: crf_dc_{}.pkl]
"""

args = docopt(USAGE)

args['<dst_name>'] = args['<dst_name>'] or 'crf_dc_{}.pkl'

param_files = glob(join(args['<param_dir>'], 'params*.pkl'))

for nid, f in enumerate(param_files):
    p = pickle.load(open(f))
    crf = mm.ml.crf.ConditionalRandomField(
        initial=p[0], final=p[1], bias=p[2], transition=p[3], observation=p[4]
    )
    crf.dump(args['<dst_name>'].format(nid + 1))
