import theano.tensor as tt
import lasagne as lnn

from . import dnn


def compute_loss(prediction, target):
    # need to clip predictions for numerical stability
    eps = 1e-7
    pred_clip = tt.clip(prediction, eps, 1.-eps)
    return lnn.objectives.binary_crossentropy(pred_clip, target).mean()


def build_net(in_shape, out_size_chroma, out_size, model):
    # first, stack the dnn chroma extractor
    chroma_network, input_var, crm_target_var = dnn.build_net(
        in_shape, out_size_chroma, model
    )

    # then, add the logistic regression chord classifier
    crd_target_var = tt.matrix('target_output', dtype='float32')

    chord_network = lnn.layers.DenseLayer(
        chroma_network, name='chords', num_units=out_size,
        nonlinearity=lnn.nonlinearities.softmax)

    # tag chord classification parameters so we can distinguish them later
    for p in chord_network.get_params():
        chord_network.params[p].add('chord')

    return (chroma_network, chord_network,
            input_var, crm_target_var, crd_target_var)


def build_model(in_shape, out_size_chroma, out_size, model):
    (crm, crd, inv, crmv, crdv) = build_net(in_shape, out_size_chroma,
                                            out_size, model)
    return dict(chroma_network=crm, chord_network=crd,
                input_var=inv, chroma_target_var=crmv, chord_target_var=crdv,
                chroma_loss_fn=compute_loss,
                chord_loss_fn=dnn.categorical_crossentropy)


create_iterators = dnn.create_iterators


def add_sacred_config(ex):

    # =============================================================== dense net

    ex.add_named_config(
        name='dense_net',
        datasource=dict(
            context_size=7,
        ),
        chroma_network=dict(
            model=dict(
                type='chroma_dnn',
                dense=dict(
                    num_layers=3,
                    num_units=512,
                    dropout=0.5,
                    nonlinearity='rectify',
                    batch_norm=False,
                ),
                out_nonlinearity='sigmoid'
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
                early_stop_acc=False,
            ),
            regularisation=dict(
                l2=1e-4,
                l1=0.0,
            ),
        ),
        optimiser=dict(
            name='adam',
            params=dict(
                learning_rate=0.001
            ),
            schedule=None
        ),
        training=dict(
            iterator='BatchIterator',
            batch_size=512,
            num_epochs=500,
            early_stop=20,
            early_stop_acc=True
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

