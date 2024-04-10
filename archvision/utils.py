
def validate_and_update_config(cfg):
    """
    asserts that the configuration is valid, and adds the default layer values.
    """
    assert cfg.nonlin in [
        "linear",
        "relu",
        "tanh",
        "sigmoid",
    ], "only linear, relu, tanh, sigmoid supported!"
    assert cfg.init in [
        "kaiming",
        "xavier",
        "gaussian",
        "uniform",
    ], "only kaiming, xavier, gaussian, uniform supported!"
    assert cfg.norm in [
        "batch",
        "instance",
        "channel",
        "none",
    ], "only batch, instance, channel, none supported!"
    default_layer = {
        "kernel_size": 3,
        "channels": 64,
        "pooling": "none",
        "pool_kernel_size": 2,
    }
    for i, layer in enumerate(cfg.model.layers):
        cfg.model.layers[i] = {**default_layer, **layer}

    return cfg
