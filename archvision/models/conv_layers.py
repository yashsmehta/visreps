import torch.nn as nn
from models.custom_operations.norm import DivNorm


class ConvolutionLayers(nn.Module):
    def __init__(self, cfg, in_channels, device):
        super(ConvolutionLayers, self).__init__()

        self.conv_layers = nn.ModuleList()
        self.n_channels = (
            in_channels
            if cfg.model.conv_in_channels == "none"
            else cfg.model.conv_in_channels
        )
        self.device = device

        for i, layer in enumerate(cfg.model.layers):
            conv_layer = nn.Conv2d(
                in_channels, layer.channels, kernel_size=layer.kernel_size, padding=1
            ).to(self.device)
            self.initialize_weights(conv_layer, cfg.model.weights_init)

            nonlinearity = self.get_nonlinearity(cfg.model.nonlin)
            normalization = self.get_normalization(layer.channels, cfg.model.norm)
            pooling = self.get_pooling(layer.pooling, layer.pool_kernel_size)
            if i == len(cfg.model.layers) - 1:
                conv_layer.__class__.__name__ = "last_layer"

            self.conv_layers.append(
                nn.Sequential(conv_layer, nonlinearity, normalization, pooling)
            )
            in_channels = layer.channels

    def forward(self, x):
        x = x.to(self.device)
        for layer in self.conv_layers:
            x = layer(x)
        return x

    def get_normalization(self, out_channels, norm_type):
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

    def get_nonlinearity(self, nonlin_type):
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

    def get_pooling(self, pool_type, pool_kernel_size):
        match pool_type:
            case "max":
                return nn.MaxPool2d(kernel_size=pool_kernel_size)
            case "avg":
                return nn.AvgPool2d(kernel_size=pool_kernel_size)
            case "none":
                return nn.Identity()
            case _:
                raise ValueError(f"Unsupported pool type: {pool_type}")

    def initialize_weights(self, conv_layer, initialization):
        match initialization:
            case "xavier":
                nn.init.xavier_normal_(conv_layer.weight)
            case "kaiming":
                nn.init.kaiming_normal_(conv_layer.weight)
            case "gaussian":
                nn.init.normal_(conv_layer.weight, mean=0, std=0.02)
            case "uniform":
                nn.init.uniform_(conv_layer.weight, a=-0.02, b=0.02)
            case _:
                raise ValueError(f"Unsupported initialization method: {initialization}")
