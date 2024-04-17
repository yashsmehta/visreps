from omegaconf import OmegaConf


def check_and_update_config(cfg):
    """
    asserts that the configuration is valid, and adds the default layer values.
    """
    assert cfg.model.name in [
        "vgg16",
        "vgg19",
        "resnet50",
        "densenet121",
        "custom",
    ], "only vgg16, vgg19, resnet50, densenet121, custom supported!"
    assert cfg.model.deepjuice_keep_layer in [
        "last_layer",
        "all",
    ], "only last_layer, all supported for deepjuice keep layer!"
    assert cfg.model.pretrained in [
        True,
        False,
    ], "pretrained must be True or False!"

    if cfg.model.name == "custom":
        assert (
            cfg.model.pretrained == False
        ), "pretrained is not supported for custom model!"
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
    else:
        cfg.model = OmegaConf.create(
            {
                "name": cfg.model.name,
                "pretrained": cfg.model.pretrained,
                "deepjuice_keep_layer": cfg.model.deepjuice_keep_layer,
            }
        )

    return cfg
