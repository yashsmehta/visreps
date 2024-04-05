from deepjuice.model_zoo.options import get_deepjuice_model


def builder(cfg):
    """
    this function will make a model based on the passed config object. It can be of 2 types, it can be a pre-determined model 
    from pytorch_models.models.py or it can be a newly created model based on the architecture parameters in config.
    """
    model, _ = get_deepjuice_model(cfg.model_uid)
    return model


def preprocess(cfg):
    _, preprocess = get_deepjuice_model(cfg.model_uid)
    return preprocess
