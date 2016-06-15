import theano.tensor as tt
import lasagne as lnn

import dmgr

from .. import augmenters
from . import blocks


def categorical_crossentropy(prediction, target):
    # need to clip predictions for numerical stability
    eps = 1e-7
    pred_clip = tt.clip(prediction, eps, 1.-eps)
    return lnn.objectives.categorical_crossentropy(pred_clip, target).mean()


def categorical_mse(predictions, targets):
    """ Mean squared error on class targets """
    return tt.mean(
        (1.0 - predictions[tt.arange(targets.shape[0]), targets]) ** 2)


def build_net(in_shape, out_size, model):
    # input variables
    input_var = (tt.tensor3('input', dtype='float32')
                 if len(in_shape) > 1 else
                 tt.matrix('input', dtype='float32'))
    target_var = tt.matrix('target_output', dtype='float32')

    # stack more layers
    network = lnn.layers.InputLayer(
        name='input', shape=(None,) + in_shape, input_var=input_var)

    if 'conv' in model and model['conv']:
        # reshape to 1 "color" channel
        network = lnn.layers.reshape(
            network, shape=(-1, 1) + in_shape, name='reshape')

        for c in sorted(model['conv'].keys()):
            network = blocks.conv(network, **model['conv'][c])

    # no more output layer if gap is already there!
    if 'gap' in model and model['gap']:
        network = blocks.gap(network, out_size=out_size,
                             out_nonlinearity=model['out_nonlinearity'],
                             **model['gap'])
    else:
        if 'dense' in model and model['dense']:
            network = blocks.dense(network, **model['dense'])

        # output layer
        out_nl = getattr(lnn.nonlinearities, model['out_nonlinearity'])
        network = lnn.layers.DenseLayer(
            network, name='output', num_units=out_size,
            nonlinearity=out_nl)

    return network, input_var, target_var


def train_iterator(train_set, training):
    it = training.get('iterator', 'BatchIterator')

    if it == 'BatchIterator':
        return dmgr.iterators.BatchIterator(
            train_set, training['batch_size'], randomise=True,
            expand=True
        )
    elif it == 'ClassBalancedIterator':
        return dmgr.iterators.UniformClassIterator(
            train_set, training['batch_size']
        )
    else:
        raise ValueError('Unknown Batch Iterator: {}'.format(it))


def build_model(in_shape, out_size, model):
    network, input_var, target_var = build_net(in_shape, out_size, model)
    return dict(network=network, input_var=input_var, target_var=target_var,
                loss_fn=categorical_crossentropy)


def create_iterators(train_set, val_set, training, augmentation):
    train_batches = train_iterator(train_set, training)
    val_batches = dmgr.iterators.BatchIterator(
        val_set, training['batch_size'], randomise=False, expand=True
    )

    if augmentation is not None:
        train_batches = dmgr.iterators.AugmentedIterator(
            train_batches, *augmenters.create_augmenters(augmentation)
        )

    return train_batches, val_batches


def add_sacred_config(ex):

    # =============================================================== dense net

    ex.add_named_config(
        name='dense_net',
        datasource=dict(
            context_size=7,
        ),
        model=dict(
            type='dnn',
            dense=dict(
                num_layers=3,
                num_units=512,
                nonlinearity='rectify',
                batch_norm=False,
                dropout=0.5,
            ),
            out_nonlinearity='softmax'
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
            batch_size=512,
            num_epochs=500,
            early_stop=20,
            early_stop_acc=True,
        ),
        regularisation=dict(
            l2=1e-4,
            l1=0.0,
        ),
        testing=dict(
            test_on_val=False,
            batch_size=None
        )
    )

    # ================================================================ conv net

    ex.add_named_config(
        name='conv_net',
        datasource=dict(
            context_size=7,
        ),
        model=dict(
            type='dnn',
            conv=dict(
                conv1=dict(
                    num_layers=4,
                    num_filters=32,
                    filter_size=(3, 3),
                    pool_size=(1, 2),
                    dropout=0.5,
                    pad='same',
                    batch_norm=True,
                ),
                conv2=dict(
                    num_layers=2,
                    num_filters=64,
                    filter_size=(3, 3),
                    pool_size=(1, 2),
                    dropout=0.5,
                    pad='valid',
                    batch_norm=True,
                ),
                conv3=dict(
                    num_layers=1,
                    num_filters=128,
                    filter_size=(9, 12),
                    pool_size=None,
                    dropout=0.5,
                    pad='valid',
                    batch_norm=True
                )
            ),
            out_nonlinearity='softmax'
        ),
        optimiser=dict(
            name='adam',
            params=dict(
                learning_rate=0.001
            ),
            schedule=None
        ),
        training=dict(
            num_epochs=500,
            early_stop=5,
            early_stop_acc=True,
            batch_size=512,
        ),
        regularisation=dict(
            l2=1e-7,
            l1=0
        ),
        testing=dict(
            test_on_val=False,
            batch_size=512
        )
    )

    @ex.named_config
    def dense_classifier():
        model = dict(
            dense=dict(
                num_layers=1,
                num_units=512,
                dropout=0.5,
                nonlinearity='rectify',
                batch_norm=False
            )
        )

    @ex.named_config
    def gap_classifier():
        model = dict(
            gap=dict(
                batch_norm=True,
                gap_nonlinearity='linear',
            )
        )




