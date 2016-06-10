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


def recurrent(network, mask_in, num_rec_units, num_layers, dropout,
              bidirectional, nonlinearity):

    if nonlinearity != 'LSTM':
        nl = getattr(lnn.nonlinearities, nonlinearity)

        def add_layer(prev_layer, **kwargs):
            return lnn.layers.RecurrentLayer(
                prev_layer, num_units=num_rec_units, mask_input=mask_in,
                nonlinearity=nl,
                W_in_to_hid=lnn.init.GlorotUniform(),
                W_hid_to_hid=lnn.init.Orthogonal(gain=np.sqrt(2) / 2),
                **kwargs)

    else:
        def add_layer(prev_layer, **kwargs):
            return lnn.layers.LSTMLayer(
                prev_layer, num_units=num_rec_units, mask_input=mask_in,
                **kwargs
            )

    fwd = network
    for i in range(num_layers):
        fwd = add_layer(fwd, name='rec_fwd_{}'.format(i))
        if dropout > 0.:
            fwd = lnn.layers.DropoutLayer(fwd, p=dropout)

    if not bidirectional:
        return network

    bck = network
    for i in range(num_layers):
        bck = add_layer(bck, name='rec_bck_{}'.format(i), backwards=True)
        if dropout > 0:
            bck = lnn.layers.DropoutLayer(bck, p=dropout)

    # combine the forward and backward recurrent layers...
    network = lnn.layers.ConcatLayer([fwd, bck], name='fwd + bck', axis=-1)
    return network

