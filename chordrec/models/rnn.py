import theano.tensor as tt

import dmgr
import lasagne as lnn

from .. import augmenters
from . import blocks


def compute_loss(prediction, target, mask):
    # need to clip predictions for numerical stability
    eps = 1e-7
    pred_clip = tt.clip(prediction, eps, 1. - eps)
    loss = lnn.objectives.categorical_crossentropy(pred_clip, target)
    return lnn.objectives.aggregate(loss, mask, mode='normalized_sum')


def build_net(in_shape, out_size, model):
    # input variables
    input_var = tt.tensor3('input', dtype='float32')
    target_var = tt.tensor3('target_output', dtype='float32')
    mask_var = tt.matrix('mask_input', dtype='float32')

    # stack more layers
    network = lnn.layers.InputLayer(
        name='input', shape=(None, None) + in_shape,
        input_var=input_var
    )

    true_batch_size, true_seq_len, _ = input_var.shape

    mask_in = lnn.layers.InputLayer(name='mask',
                                    input_var=mask_var,
                                    shape=(None, None))

    network = blocks.recurrent(network, mask_in, **model['recurrent'])

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

    return network, input_var, target_var, mask_var


def build_model(in_shape, out_size, model):
    network, input_var, target_var, mask_var = build_net(in_shape, out_size,
                                                         model)
    return dict(network=network, input_var=input_var, target_var=target_var,
                mask_var=mask_var, loss_fn=compute_loss)


def create_iterators(train_set, val_set, training, augmentation):
    train_batches = dmgr.iterators.SequenceIterator(
        train_set, training['batch_size'], randomise=True,
        expand=True, max_seq_len=training['max_seq_len']
    )

    val_batches = dmgr.iterators.SequenceIterator(
        val_set, training['batch_size'], randomise=False,
        expand=False
    )

    if augmentation is not None:
        train_batches = dmgr.iterators.AugmentedIterator(
            train_batches, *augmenters.create_augmenters(augmentation)
        )

    return train_batches, val_batches


def add_sacred_config(ex):
    ex.add_named_config(
        name='recurrent',
        model=dict(
            type='rnn',
            recurrent=dict(
                num_rec_units=128,
                num_layers=3,
                dropout=0.3,
                bidirectional=True,
                nonlinearity='rectify'
            )
        ),
        optimiser=dict(
            name='adam',
            params=dict(
                learning_rate=0.0001
            ),
            schedule=None
        ),
        training=dict(
            iterator='BatchIterator',
            batch_size=8,
            max_seq_len=64,
            num_epochs=1000,
            early_stop=20,
            early_stop_acc=True,
        ),
        regularisation=dict(
            l1=0.0,
            l2=1e-4,
        ),
        testing=dict(
            test_on_val=False
        )
    )

    @ex.named_config
    def lstm():
        net = dict(
            nonlinearity='LSTM',
            num_rec_units=64,
        )
