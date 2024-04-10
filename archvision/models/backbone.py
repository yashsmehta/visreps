import torch.nn as nn
import torchvision.models as models
from .conv_layers import ConvolutionLayers
# from .wavelet_layers import WaveletLayers


class VisionModel(nn.Module):
    def __init__(self, cfg, device):
        super(VisionModel, self).__init__()
        self.device = device
        # self.wavelet_layers = WaveletLayers(cfg)
        self.conv_layers = ConvolutionLayers(cfg, self.device)

    def forward(self, x):
        x = x.to(self.device)
        # x = self.wavelet_layers(x)
        x = self.conv_layers(x)
        return x


def AlexNet():
    alexnet = models.alexnet(weights="AlexNet_Weights.IMAGENET1K_V1")
    return alexnet
