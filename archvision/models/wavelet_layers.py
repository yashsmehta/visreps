import torch.nn as nn
import os
from models.custom_operations.convolution import WaveletConvolution
from models.custom_operations.norm import DivNorm


class WaveletLayers(nn.Module):
    def __init__(self, cfg, device):
        super(WaveletLayers, self).__init__()

        self.wavelet_layers = nn.ModuleList()
        self.device = device

        wavelet_layer = WaveletConvolution(
            filter_size=15,
            filter_type="curvature",
            filter_params=cfg.model.wavelet,
            device=self.device,
        ).to(self.device)
        self.out_channels = wavelet_layer.layer_size
        print("wavelet layer size: ", self.out_channels)

        nonlinearity = self.get_nonlinearity(cfg.model.nonlin)
        pooling = self.get_pooling("avg", 2)

        self.wavelet_layers.append(nn.Sequential(wavelet_layer, nonlinearity, pooling))

    def forward(self, x):
        x = x.to(self.device)
        for layer in self.wavelet_layers:
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
            case "layer":
                return nn.LayerNorm(out_channels)
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
