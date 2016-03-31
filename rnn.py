import numpy as np
import theano.tensor as tt
import lasagne as lnn
from exp_utils import setup_experiment, run_experiment


def compute_loss(prediction, target, mask):
    # need to clip predictions for numerical stability
    eps = 1e-7
    pred_clip = tt.clip(prediction, eps, 1. - eps)
    loss = lnn.objectives.categorical_crossentropy(pred_clip, target)
    return lnn.objectives.aggregate(loss, mask, mode='normalized_sum')


def stack_layers(net, mask_in, num_rec_units, num_layers, dropout, grad_clip,
                 bidirectional, nonlinearity):

    if nonlinearity != 'LSTM':
        nl = getattr(lnn.nonlinearities, nonlinearity)

        def add_layer(prev_layer, **kwargs):
            return lnn.layers.RecurrentLayer(
                prev_layer, num_units=num_rec_units, mask_input=mask_in,
                grad_clipping=grad_clip, nonlinearity=nl,
                W_in_to_hid=lnn.init.GlorotUniform(),
                W_hid_to_hid=lnn.init.Orthogonal(gain=np.sqrt(2) / 2),
                **kwargs)

    else:
        def add_layer(prev_layer, **kwargs):
            return lnn.layers.LSTMLayer(
                prev_layer, num_units=num_rec_units, mask_input=mask_in,
                grad_clipping=grad_clip,
                **kwargs
            )

    fwd = net
    for i in range(num_layers):
        fwd = add_layer(fwd, name='rec_fwd_{}'.format(i))
        if dropout > 0.:
            fwd = lnn.layers.DropoutLayer(fwd, p=dropout)

    if not bidirectional:
        return net

    bck = net
    for i in range(num_layers):
        bck = add_layer(bck, name='rec_bck_{}'.format(i), backwards=True)
        if dropout > 0:
            bck = lnn.layers.DropoutLayer(bck, p=dropout)

    # combine the forward and backward recurrent layers...
    net = lnn.layers.ConcatLayer([fwd, bck], name='fwd + bck', axis=-1)
    return net


def build_net(feature_shape, out_size, net):
    # input variables
    input_var = tt.tensor3('input', dtype='float32')
    target_var = tt.tensor3('target_output', dtype='float32')
    mask_var = tt.matrix('mask_input', dtype='float32')

    # stack more layers
    network = lnn.layers.InputLayer(
        name='input', shape=(None, None) + feature_shape,
        input_var=input_var
    )

    true_batch_size, true_seq_len, _ = input_var.shape

    mask_in = lnn.layers.InputLayer(name='mask',
                                    input_var=mask_var,
                                    shape=(None, None))

    network = stack_layers(network, mask_in, **net)

    # In order to connect a recurrent layer to a dense layer, we need to
    # flatten the first two dimensions (our "sample dimensions"); this will
    # cause each time step of each sequence to be processed independently
    network = lnn.layers.ReshapeLayer(
        network, (-1, lnn.layers.get_output_shape(network)[-1]),
        name='reshape to single')

    network = lnn.layers.DenseLayer(
        network, num_units=out_size, nonlinearity=lnn.nonlinearities.softmax,
        name='output')

    # To reshape back to our original shape, we can use the symbolic shape
    # variables we retrieved above.
    network = lnn.layers.ReshapeLayer(
        network, (true_batch_size, true_seq_len, out_size),
        name='output-reshape')

    return network, input_var, mask_var, target_var


# Initialise Sacred experiment
ex = setup_experiment('Recurrent Neural Network')
run_exp = ex.capture(run_experiment)


@ex.config
def config():
    observations = 'results'

    feature_extractor = None

    target = None

    net = dict(
        num_rec_units=128,
        num_layers=3,
        dropout=0.3,
        grad_clip=0,
        bidirectional=True,
        nonlinearity='rectify'
    )

    optimiser = dict(
        name='adam',
        params=dict(
            learning_rate=0.0001
        ),
        schedule=None
    )

    training = dict(
        num_epochs=1000,
        early_stop=20,
        early_stop_acc=True,
        batch_size=8,
        max_seq_len=64
    )

    regularisation = dict(
        l1=0.0,
        l2=1e-8,
    )

    testing = dict(
        test_on_val=False
    )


@ex.named_config
def lstm():
    net = dict(
        nonlinearity='LSTM',
        num_rec_units=64,
    )


@ex.automain
def main():
    run_exp(ex, build_fn=build_net, loss_fn=compute_loss)
