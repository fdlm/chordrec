from dnn import *


def build_model(in_shape, out_size, model):
    network, input_var, target_var = build_net(in_shape, out_size, model)

    # this goes back to the nonlinearity layer of the penultimate conv layer
    # (after batchnorm!)
    feature_layer = network
    for _ in range(7):
        feature_layer = feature_layer.input_layer

    # average the feature maps of this conv layer
    feature_out = lnn.layers.get_output(feature_layer, deterministic=True)
    feature_out = tt.mean(feature_out, axis=(2, 3))

    return dict(network=network, input_var=input_var, target_var=target_var,
                loss_fn=categorical_crossentropy, feature_out=feature_out)


def add_sacred_config(ex):
    # ======================================================= conv net with gap

    ex.add_named_config(
        name='gap_feature_extractor',
        datasource=dict(
            context_size=7,
        ),
        model=dict(
            type='avg_gap_feature',
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
            gap=dict(
                batch_norm=True,
                gap_nonlinearity='linear',
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

    ex.add_named_config(
        name='gap_feature_extractor_mm_2016',
        datasource=dict(
            context_size=11,
        ),
        model=dict(
            type='avg_gap_feature',
            conv=dict(
                conv1=dict(
                    num_layers=4,
                    num_filters=32,
                    filter_size=(3, 3),
                    pool_size=(1, 2),
                    dropout=0.5,
                    pad='valid',
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
            gap=dict(
                batch_norm=True,
                gap_nonlinearity='linear',
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
