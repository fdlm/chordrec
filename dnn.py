import theano.tensor as tt
import lasagne as lnn
import experiment


def compute_loss(prediction, target):
    # need to clip predictions for numerical stability
    eps = 1e-7
    pred_clip = tt.clip(prediction, eps, 1.-eps)
    return lnn.objectives.categorical_crossentropy(pred_clip, target).mean()


def stack_layers(net, batch_norm, nonlinearity, num_layers, num_units,
                 dropout):

    nl = getattr(lnn.nonlinearities, nonlinearity)

    for i in range(num_layers):
        net = lnn.layers.DenseLayer(
            net, num_units=num_units, nonlinearity=nl,
            name='fc-{}'.format(i)
        )
        if batch_norm:
            net = lnn.layers.batch_norm(net)
        net = lnn.layers.DropoutLayer(net, p=dropout)

    return net


def build_net(feature_shape, out_size, net):
    # input variables
    input_var = (tt.tensor3('input', dtype='float32')
                 if len(feature_shape) > 1 else
                 tt.matrix('input', dtype='float32'))
    target_var = tt.matrix('target_output', dtype='float32')

    # stack more layers
    network = lnn.layers.InputLayer(
        name='input', shape=(None,) + feature_shape, input_var=input_var)

    network = stack_layers(network, **net)

    # output layer
    network = lnn.layers.DenseLayer(
        network, name='output', num_units=out_size,
        nonlinearity=lnn.nonlinearities.softmax)

    return network, input_var, target_var


# Initialise Sacred experiment
ex = experiment.setup('Deep Neural Network')
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
        num_layers=3,
        num_units=512,
        nonlinearity='rectify',
        batch_norm=False,
        dropout=0.5,
    )

    optimiser = dict(
        name='adam',
        params=dict(
            learning_rate=0.0001
        ),
        schedule=None
    )

    training = dict(
        num_epochs=500,
        early_stop=20,
        batch_size=512,
        early_stop_acc=True,
    )

    regularisation = dict(
        l2=1e-4,
        l1=0.0,
    )

    testing = dict(
        test_on_val=False
    )


@ex.named_config
def learn_rate_schedule():
    optimiser = dict(
        schedule=dict(
            interval=10,
            factor=0.5
        )
    )


@ex.named_config
def no_context():
    datasource = dict(
        context_size=0
    )

    net = dict(
        num_units=100,
        dropout=0.3,
        l2=0.
    )


@ex.automain
def main():
    run_exp(ex, build_fn=build_net, loss_fn=compute_loss)
