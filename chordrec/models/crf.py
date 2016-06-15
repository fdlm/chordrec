import theano.tensor as tt

import dmgr
import lasagne as lnn
import spaghetti as spg

from .. import augmenters


class CrfLoss:

    def __init__(self, crf):
        self.crf = crf

    def __call__(self, prediction, target, mask):
        loss = spg.objectives.neg_log_likelihood(self.crf, target, mask)
        loss /= mask.sum(axis=1)  # normalise to sequence length
        return lnn.objectives.aggregate(loss, mode='mean')


def build_net(in_shape, out_size, model):
    # input variables
    input_var = (tt.tensor4('input', dtype='float32')
                 if len(in_shape) > 1 else
                 tt.tensor3('input', dtype='float32'))
    target_var = tt.tensor3('target_output', dtype='float32')
    mask_var = tt.matrix('mask_input', dtype='float32')

    # stack more layers
    network = lnn.layers.InputLayer(
        name='input', shape=(None, None) + in_shape,
        input_var=input_var
    )

    mask_in = lnn.layers.InputLayer(name='mask',
                                    input_var=mask_var,
                                    shape=(None, None))

    network = spg.layers.CrfLayer(
        network, mask_input=mask_in, num_states=out_size, name='CRF')

    return network, input_var, target_var, mask_var


def build_model(in_shape, out_size, model):
    network, input_var, target_var, mask_var = build_net(in_shape, out_size,
                                                         model)
    loss_fn = CrfLoss(network)
    return dict(network=network, input_var=input_var, target_var=target_var,
                mask_var=mask_var, loss_fn=loss_fn)


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
        name='crf',
        datasource=dict(
            context_size=0,
        ),
        model=dict(
            type='crf'
        ),
        optimiser=dict(
            name='adam',
            params=dict(
                learning_rate=0.01
            ),
            schedule=None
        ),
        training=dict(
            batch_size=32,
            max_seq_len=1024,
            num_epochs=500,
            early_stop=20,
            early_stop_acc=True,
        ),
        regularisation=dict(
            l1=1e-4,
            l2=0.0,
        ),
        testing=dict(
            test_on_val=False,
            batch_size=None,
        )
    )
