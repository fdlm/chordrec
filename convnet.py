import theano.tensor as tt
import lasagne as lnn
import dnn
import experiment


def compute_loss(prediction, target):
    # need to clip predictions for numerical stability
    eps = 1e-7
    pred_clip = tt.clip(prediction, eps, 1.-eps)
    return lnn.objectives.categorical_crossentropy(pred_clip, target).mean()


def stack_layers(net, batch_norm, conv1, conv2, conv3):

    for i, conv in enumerate([conv1, conv2, conv3]):
        if not conv:
            continue
        for k in range(conv['num_layers']):
            net = lnn.layers.Conv2DLayer(
                net, num_filters=conv['num_filters'],
                filter_size=conv['filter_size'],
                nonlinearity=lnn.nonlinearities.rectify,
                name='Conv_{}_{}'.format(i, k))
            if batch_norm:
                net = lnn.layers.batch_norm(net)

        net = lnn.layers.MaxPool2DLayer(net, pool_size=conv['pool_size'],
                                        name='Pool_{}'.format(i))
        net = lnn.layers.DropoutLayer(net, p=conv['dropout'])

    return net


def stack_gap(net, out_size, num_filters, filter_size, dropout, batch_norm):
    net = lnn.layers.Conv2DLayer(
        net, num_filters=num_filters, filter_size=filter_size,
        pad=0, nonlinearity=lnn.nonlinearities.rectify,
        name='Gap_Filters')
    if batch_norm:
        net = lnn.layers.batch_norm(net)

    net = lnn.layers.DropoutLayer(net, p=dropout)

    net = lnn.layers.Conv2DLayer(
        net, num_filters=num_filters, filter_size=1,
        pad=0, nonlinearity=lnn.nonlinearities.rectify,
        name='Gap_Filters_Single')
    if batch_norm:
        net = lnn.layers.batch_norm(net)
    net = lnn.layers.DropoutLayer(net, p=dropout)

    # output classification layer
    net = lnn.layers.Conv2DLayer(
        net, num_filters=out_size, filter_size=1,
        nonlinearity=lnn.nonlinearities.rectify, name='Output_Conv')
    if batch_norm:
        net = lnn.layers.batch_norm(net)

    net = lnn.layers.Pool2DLayer(
        net, pool_size=net.output_shape[-2:], ignore_border=False,
        mode='average_exc_pad', name='GlobalAveragePool')
    net = lnn.layers.FlattenLayer(net, name='Flatten')
    net = lnn.layers.NonlinearityLayer(
        net, nonlinearity=lnn.nonlinearities.softmax, name='output')

    return net


def build_net(feature_shape, out_size, net):

    if net['dense'] and net['global_avg_pool']:
        raise ValueError('Cannot use dense layers AND global avg. pool.')

    # input variables
    input_var = tt.tensor3('feature_input', dtype='float32')
    target_var = tt.matrix('target_output', dtype='float32')

    # stack more layers
    network = lnn.layers.InputLayer(
        name='input', shape=(None,) + feature_shape, input_var=input_var)

    # reshape to 1 "color" channel
    network = lnn.layers.reshape(
        network, shape=(-1, 1) + feature_shape, name='reshape')

    network = stack_layers(network, **net['conv'])

    if net['dense']:
        network = dnn.stack_layers(network, **net['dense'])
        # output classification layer
        network = lnn.layers.DenseLayer(
            network, name='output', num_units=out_size,
            nonlinearity=lnn.nonlinearities.softmax)
    elif net['global_avg_pool']:
        network = stack_gap(network, out_size, **net['global_avg_pool'])
    else:
        raise RuntimeError('Need to specify output architecture!')

    return network, input_var, target_var


# Initialise Sacred experiment
ex = experiment.setup('Convolutional Neural Network')
run_exp = ex.capture(experiment.run)


@ex.config
def config():
    observations = 'results'

    datasource = dict(
            context_size=7,
    )

    feature_extractor = None

    target = None

    net = dict(
        conv=dict(
            batch_norm=False,
            conv1=dict(
                num_layers=2,
                num_filters=32,
                filter_size=(3, 3),
                pool_size=(1, 2),
                dropout=0.5,
            ),
            conv2=dict(
                num_layers=1,
                num_filters=64,
                filter_size=(3, 3),
                pool_size=(1, 2),
                dropout=0.5,
            ),
            conv3={},
        ),
        global_avg_pool=None,
        dense=dict(
            num_layers=1,
            num_units=512,
            dropout=0.5,
            nonlinearity='rectify',
            batch_norm=False
        ),
    )

    optimiser = dict(
        name='adam',
        params=dict(
            learning_rate=0.001
        ),
        schedule=None
    )

    training = dict(
        num_epochs=500,
        early_stop=20,
        early_stop_acc=True,
        batch_size=512,
    )

    regularisation = dict(
        l2=1e-4,
        l1=0
    )

    testing = dict(
        test_on_val=False
    )


@ex.named_config
def third_conv_layer():
    net = dict(
        conv3=dict(
            num_layers=1,
            num_filters=64,
            filter_size=(3, 3),
            pool_size=(1, 2),
            dropout=0.5,
        )
    )


@ex.named_config
def gap_classifier():
    net = dict(
        dense=None,
        global_avg_pool=dict(
            num_filters=512,
            filter_size=(3, 3),
            dropout=0.5,
            batch_norm=True
        )
    )


@ex.named_config
def learn_rate_schedule():
    optimiser = dict(
        schedule=dict(
            interval=10,
            factor=0.5
        )
    )


@ex.automain
def main():
    run_exp(ex, build_fn=build_net, loss_fn=compute_loss)
