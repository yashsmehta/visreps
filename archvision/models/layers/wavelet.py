import torch.nn as nn
from archvision.models.custom_operations.wavelet_conv import WaveletConvolution
import models.nn_ops as nn_ops


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

        nonlinearity = nn_ops.get_nonlinearity(cfg.model.nonlin)
        pooling = nn_ops.get_pooling("max", 2)

        self.wavelet_layers.append(nn.Sequential(wavelet_layer, nonlinearity, pooling))

    def forward(self, x):
        x = x.to(self.device)
        for layer in self.wavelet_layers:
            x = layer(x)
        return x
