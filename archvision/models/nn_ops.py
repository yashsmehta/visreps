import torch
import torch.nn as nn
from archvision.models.custom_operations.norm import DivNorm


def get_normalization(out_channels, norm_type):
    """
    Retrieve a normalization layer based on the specified type.

    Args:
        out_channels (int): The number of output channels for the normalization layer.
        norm_type (str): The type of normalization to apply. Supported types are "batch", "channel",
                         "instance", and "none". An unsupported type raises a ValueError.

    Returns:
        torch.nn.Module: A normalization layer corresponding to the specified type.

    Raises:
        ValueError: If an unsupported normalization type is provided.
    """
    match norm_type:
        case "batch":
            return nn.BatchNorm2d(out_channels)
        case "channel":
            return DivNorm()
        case "instance":
            return nn.InstanceNorm2d(out_channels)
        case "none":
            return nn.Identity()
        case _:
            raise ValueError(f"Unsupported normalization method: {norm_type}")

def get_nonlinearity(nonlin_type, inplace=True):
    """
    Retrieve a nonlinearity activation function based on the specified type.

    Args:
        nonlin_type (str): The type of nonlinearity to use. Supported types are "relu", "tanh",
                           "sigmoid", "elu", and "none". An unsupported type raises a ValueError.
        inplace (bool): Whether the operation should be performed in-place. Default is True.

    Returns:
        torch.nn.Module: A nonlinearity activation function corresponding to the specified type.

    Raises:
        ValueError: If an unsupported non-linearity type is provided.
    """
    match nonlin_type:
        case "relu":
            return nn.ReLU(inplace=inplace)
        case "tanh":
            return nn.Tanh()
        case "sigmoid":
            return nn.Sigmoid()
        case "elu":
            return nn.ELU(inplace=inplace)
        case "none":
            return nn.Identity()
        case _:
            raise ValueError(f"Unsupported non-linearity: {nonlin_type}")

def get_pooling(pool_type, pool_kernel_size):
    """
    Retrieve a pooling layer based on the specified type and kernel size.

    Args:
        pool_type (str): The type of pooling to use. Supported types are "max", "avg", "globalmax",
                         "globalavg", and "none". An unsupported type raises a ValueError.
        pool_kernel_size (int or tuple): The size of the pooling kernel. Not applicable for global pooling types.

    Returns:
        torch.nn.Module: A pooling layer corresponding to the specified type and kernel size.

    Raises:
        ValueError: If an unsupported pool type is provided.
    """
    match pool_type:
        case "max":
            return nn.MaxPool2d(kernel_size=pool_kernel_size)
        case "avg":
            return nn.AvgPool2d(kernel_size=pool_kernel_size)
        case "globalmax":
            return nn.AdaptiveMaxPool2d((1, 1))
        case "globalavg":
            return nn.AdaptiveAvgPool2d((1, 1))
        case "none":
            return nn.Identity()
        case _:
            raise ValueError(f"Unsupported pool type: {pool_type}")

def initialize_weights(conv_layer, initialization, seed):
    """
    Initialize the weights of a convolutional layer based on the specified method and seed.

    Args:
        conv_layer (torch.nn.Module): The convolutional layer whose weights are to be initialized.
        initialization (str): The method of weight initialization. Supported methods are "xavier",
                              "xavier_uniform", "kaiming", "kaiming_uniform", "gaussian", and "uniform".
                              An unsupported method raises a ValueError.
        seed (int): The seed for random number generation to ensure reproducibility.

    Returns:
        None: The weights of the convolutional layer are modified in-place.

    Raises:
        ValueError: If an unsupported initialization method is provided.
    """
    torch.manual_seed(seed)
    match initialization:
        case "xavier":
            nn.init.xavier_normal_(conv_layer.weight)
        case "xavier_uniform":
            nn.init.xavier_uniform_(conv_layer.weight)
        case "kaiming":
            nn.init.kaiming_normal_(conv_layer.weight)
        case "kaiming_uniform":
            nn.init.kaiming_uniform_(conv_layer.weight)
        case "gaussian":
            nn.init.normal_(conv_layer.weight, mean=0, std=0.02)
        case "uniform":
            nn.init.uniform_(conv_layer.weight, a=-0.02, b=0.02)
        case _:
            raise ValueError(f"Unsupported initialization method: {initialization}")
