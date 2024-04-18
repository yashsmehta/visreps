from omegaconf import OmegaConf


def check_and_update_config(cfg):
    """
    asserts that the configuration is valid, and adds the default layer values.
    """
    assert cfg.model.name in [
        "alexnet",
        "vgg16",
        "resnet50",
        "densenet121",
        "custom",
    ], "only vgg16, resnet50, densenet121, custom supported!"
    assert cfg.model.deepjuice_keep_layer in [
        "last_layer",
        "all",
    ], "only last_layer, all supported for deepjuice keep layer!"
    assert cfg.model.pretrained in [
        True,
        False,
    ], "pretrained must be True or False!"

    if cfg.model.name == "custom":
        cfg.model.pretrained = False
        assert cfg.model.nonlin in [
            "none",
            "relu",
            "tanh",
            "sigmoid",
        ], "only linear, relu, tanh, sigmoid supported!"
        assert cfg.model.weights_init in [
            "kaiming",
            "kaiming_uniform",
            "xavier",
            "gaussian",
            "uniform",
        ], "only kaiming, xavier, gaussian, uniform supported!"
        assert cfg.model.norm in [
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
            default_layer.update(layer)
            cfg.model.layers[i] = default_layer.copy()

            assert cfg.model.layers[i].pooling in [
                "max",
                "globalmax",
                "avg",
                "globalavg",
                "none",
            ], "only max, avg, globalmax, globalavg and none supported!"
            if layer.pooling in ["globalmax", "globalavg"]:
                cfg.model.layers[i].pool_kernel_size = "N/A"
    else:
        cfg.model = OmegaConf.create(
            {
                "name": cfg.model.name,
                "pretrained": cfg.model.pretrained,
                "deepjuice_keep_layer": cfg.model.deepjuice_keep_layer,
            }
        )

    return cfg
