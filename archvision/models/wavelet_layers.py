import torch.nn as nn
import torch
from models.custom_operations.convolution import WaveletConvolution


class WaveletLayers(nn.Module):
    #TODO:!!
    def __init__(self, cfg, device):
        super(WaveletLayers, self).__init__()
        wavelet_conv = WaveletConvolution(filter_size=###
                                        filter_type=cfg.wavelet_type[i],
                                        device=self.device)
        out_channels = wavelet_conv.layer_size


    def forward(self, x):
        with torch.no_grad():
            for layer in self.layers:
                x = layer(x)
        return x
