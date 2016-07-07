#!/usr/bin/env python

import madmom as mm
import pickle
from docopt import docopt

USAGE = """
create_madmom_crf_model.py - creates madmom CRF models.

Usage:
    create_madmom_deep_chroma_model.py <spg_mdl> <mm_mdl>

Arguments:
    <spg_mdl>  source spaghetti model file
    <mm_mdl>   destination madmom model file
"""

args = docopt(USAGE)

pi, tau, c, A, W = pickle.load(open(args['<spg_mdl>']))
crf = mm.ml.crf.ConditionalRandomField(pi, tau, c, A, W)
pickle.dump(crf, open(args['<mm_mdl>'], 'wb'))
