import numpy as np

import lasagne as lnn


def conv(network, batch_norm, num_layers, num_filters, filter_size, pad,
         pool_size, dropout):
    for k in range(num_layers):
        network = lnn.layers.Conv2DLayer(
            network, num_filters=num_filters,
            filter_size=filter_size,
            W=lnn.init.Orthogonal(gain=np.sqrt(2 / (1 + .1 ** 2))),
            pad=pad,
            nonlinearity=lnn.nonlinearities.rectify,
            name='Conv_{}'.format(k))
        if batch_norm:
            network = lnn.layers.batch_norm(network)

    if pool_size:
        network = lnn.layers.MaxPool2DLayer(network, pool_size=pool_size,
                                            name='Pool')
    if dropout > 0.0:
        network = lnn.layers.DropoutLayer(network, p=dropout)

    return network


def gap(network, out_size, batch_norm,
        gap_nonlinearity, out_nonlinearity):

    gap_nonlinearity = getattr(lnn.nonlinearities, gap_nonlinearity)
    out_nonlinearity = getattr(lnn.nonlinearities, out_nonlinearity)

    # output classification layer
    network = lnn.layers.Conv2DLayer(
        network, num_filters=out_size, filter_size=1,
        nonlinearity=gap_nonlinearity, name='Output_Conv')
    if batch_norm:
        network = lnn.layers.batch_norm(network)

    network = lnn.layers.Pool2DLayer(
        network, pool_size=network.output_shape[-2:], ignore_border=False,
        mode='average_exc_pad', name='GlobalAveragePool')
    network = lnn.layers.FlattenLayer(network, name='Flatten')

    network = lnn.layers.NonlinearityLayer(
        network, nonlinearity=out_nonlinearity, name='output')

    return network


def dense(network, batch_norm, nonlinearity, num_layers, num_units,
          dropout):

    nl = getattr(lnn.nonlinearities, nonlinearity)

    for i in range(num_layers):
        network = lnn.layers.DenseLayer(
            network, num_units=num_units, nonlinearity=nl,
            name='fc-{}'.format(i)
        )
        if batch_norm:
            network = lnn.layers.batch_norm(network)
        if dropout > 0.0:
            network = lnn.layers.DropoutLayer(network, p=dropout)

    return network
