import torch.nn as nn
import torchvision.models as models
from .conv_layers import ConvolutionLayers
from .wavelet_layers import WaveletLayers


class VisionModel(nn.Module):
    def __init__(self, cfg, device):
        super(VisionModel, self).__init__()
        self.device = device
        self.wavelet_layers = WaveletLayers(cfg, self.device)
        self.conv_layers = ConvolutionLayers(cfg, self.wavelet_layers.out_channels, self.device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.wavelet_layers(x)
        x = self.conv_layers(x)
        return x


def AlexNet(pretrained=True):
    if pretrained:
        alexnet = models.alexnet(weights="AlexNet_Weights.IMAGENET1K_V1")
    else:
        alexnet = models.alexnet(weights=None)
    last_layer = alexnet.classifier[-1]
    last_layer.__class__.__name__ = "last_layer"

    return alexnet

def VGGNet(pretrained=True):
    if pretrained:
        vggnet = models.vgg16(weights="VGG16_Weights.IMAGENET1K_V1")
    else:
        vggnet = models.vgg16(weights=None)

    last_layer = vggnet.classifier[-1]
    last_layer.__class__.__name__ = "last_layer"
    return vggnet
