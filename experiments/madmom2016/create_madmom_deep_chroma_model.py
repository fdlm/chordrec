#!/usr/bin/env python

import madmom as mm
import pickle
from docopt import docopt

USAGE = """
create_madmom_deep_chroma_model.py - creates madmom models for the
                                     DeepChromaProcessor.

Usage:
    create_madmom_deep_chroma_model.py <cr_model> <mm_model>

Arguments:
    <cr_model>  source lasagne model
    <mm_model>  destination madmom model
"""

args = docopt(USAGE)

p = pickle.load(open(args['<cr_model>']))
nn = mm.ml.nn.NeuralNetwork([
    mm.ml.nn.layers.FeedForwardLayer(
        p[i], p[i+1],
        # relu layers, but last layer is sigmoid
        mm.ml.nn.activations.relu if i < len(p) - 4 else
        mm.ml.nn.activations.sigmoid
    )
    for i in range(0, len(p) - 2, 2)
])
nn.dump(args['<mm_model>'])
