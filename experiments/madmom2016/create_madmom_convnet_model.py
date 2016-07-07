#!/usr/bin/env python

import madmom as mm
import numpy as np
import pickle
from docopt import docopt
from madmom.ml.nn.layers import (ConvolutionalLayer,
                                 BatchNormLayer, MaxPoolLayer)
from madmom.ml.nn.activations import relu

USAGE = """
create_madmom_convnet_model.py - creates madmom convnet models for chord rec.

Usage:
    create_madmom_convnet_model.py <cr_model> <mm_model>

Arguments:
    <cr_model>  source lasagne model file name
    <mm_model>  destination madmom model file name
"""

args = docopt(USAGE)


def conv_block(p, n_layers):
    layers = []
    for i in range(n_layers):
        layers.append(ConvolutionalLayer(p[0].transpose(1, 0, 2, 3),
                                         np.array([0])))
        layers.append(BatchNormLayer(*p[1:5], activation_fn=relu))
        del p[:5]
    return layers

p = pickle.load(open(args['<cr_model>']))

layers = []
layers += conv_block(p, 4)
layers.append(MaxPoolLayer((1, 2)))
layers += conv_block(p, 2)
layers.append(MaxPoolLayer((1, 2)))
layers += conv_block(p, 1)

nn = mm.ml.nn.NeuralNetwork(layers)
nn.dump(args['<mm_model>'])
