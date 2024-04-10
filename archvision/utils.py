
def check_and_update_config(cfg):
    """
    asserts that the configuration is valid, and adds the default layer values.
    """
    assert cfg.model.nonlin in [
        "none",
        "relu",
        "tanh",
        "sigmoid",
    ], "only linear, relu, tanh, sigmoid supported!"
    assert cfg.model.weights_init in [
        "kaiming",
        "xavier",
        "gaussian",
        "uniform",
    ], "only kaiming, xavier, gaussian, uniform supported!"
    assert cfg.model.norm in [
        "batch",
        "layer",
        "instance",
        "channel",
        "none",
    ], "only batch, instance, channel, none supported!"
    default_layer = {
        "kernel_size": 3,
        "channels": 64,
        "pooling": "none",
        "pool_kernel_size": "none",
    }
    for i, layer in enumerate(cfg.model.layers):
        cfg.model.layers[i] = {**default_layer, **layer}
        assert cfg.model.layers[i].pooling in [
            "none",
            "max",
            "avg",
        ], "only None, max, avg supported!"

    return cfg
