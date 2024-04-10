import torch.nn as nn
from models.custom_operations.norm import DivNorm


class ConvolutionLayers(nn.Module):
    def __init__(self, cfg):
        super(ConvolutionLayers, self).__init__()

        self.conv_layers = nn.ModuleList()
        in_channels = cfg.in_channels

        for layer in cfg.model.layers:
            conv_layer = nn.Conv2d(in_channels, layer.channels, kernel_size=layer.kernel_size, padding=1)
            norm_layer = self.get_norm_layer(layer.channels, cfg.model.norm)
            nonlin_layer = self.get_nonlin_layer(cfg.model.nonlin)
            pool_layer = self.get_pool_layer(layer.pooling, layer.pool_kernel_size)
            
            self.conv_layers.append(nn.Sequential(conv_layer, norm_layer, nonlin_layer, pool_layer))
            in_channels = layer.channels

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        return x

    def get_norm_layer(self, out_channels, norm_type):
        match norm_type:
            case 'batch':
                return nn.BatchNorm2d(out_channels)
            case 'channel':
                return DivNorm()
            case 'instance':
                return nn.InstanceNorm2d(out_channels)
            case 'none':
                return nn.Identity()
            case _:
                raise ValueError(f"Unsupported normalization method: {norm_type}")

    def get_nonlin_layer(self, nonlin_type):
        match nonlin_type:
            case 'relu':
                return nn.ReLU(inplace=True)
            case 'tanh':
                return nn.Tanh()
            case 'sigmoid':
                return nn.Sigmoid()
            case 'elu':
                return nn.ELU(inplace=True)
            case 'none':
                return nn.Identity()
            case _:
                raise ValueError(f"Unsupported non-linearity: {nonlin_type}")

    def get_pool_layer(self, pool_type, pool_kernel_size):
        match pool_type:
            case 'max':
                return nn.MaxPool2d(kernel_size=pool_kernel_size)
            case 'avg':
                return nn.AvgPool2d(kernel_size=pool_kernel_size)
            case 'none':
                return nn.Identity()
            case _:
                raise ValueError(f"Unsupported pool type: {pool_type}")

    def initialize_weights(self, conv_layer, init_method):
        match init_method:
            case "xavier":
                nn.init.xavier_normal_(conv_layer.weight)
            case "kaiming":
                nn.init.kaiming_normal_(conv_layer.weight)
            case "gaussian":
                nn.init.normal_(conv_layer.weight, mean=0, std=0.02)
            case "uniform":
                nn.init.uniform_(conv_layer.weight, a=-0.02, b=0.02)
            case _:
                raise ValueError(f"Unsupported initialization method: {init_method}")



    