import torch
import torch.nn as nn
from models.custom_operations.norm import DivNorm


def get_normalization(out_channels, norm_type):
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

def get_nonlinearity(nonlin_type):
    match nonlin_type:
        case "relu":
            return nn.ReLU(inplace=True)
        case "tanh":
            return nn.Tanh()
        case "sigmoid":
            return nn.Sigmoid()
        case "elu":
            return nn.ELU(inplace=True)
        case "none":
            return nn.Identity()
        case _:
            raise ValueError(f"Unsupported non-linearity: {nonlin_type}")

def get_pooling(pool_type, pool_kernel_size):
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
