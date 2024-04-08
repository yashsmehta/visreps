import torch.nn as nn
from .conv_layers import ConvolutionLayers
from .wavelet_layers import WaveletLayers

class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        self.wavelet_layers = WaveletLayers()
        self.conv_layers = ConvolutionLayers()

    def forward(self, x):
        x = self.wavelet_model(x)
        x = self.conv_net(x)
        
        return x