import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, AlexNet_Weights, VGG16_Weights, DenseNet121_Weights
from .conv_layers import ConvolutionLayers
from .wavelet_layers import WaveletLayers
from .last_layer import Last
import torch

class EncapsulatedVisionModel:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.model = VisionModel(self.cfg, self.device).to(self.device)

    def forward(self, x):
        return self.model(x)


class TestModel(nn.Module):
    def __init__(self, cfg, device):
        super(TestModel, self).__init__()
        self.cfg = cfg
        self.device = device
        # Encapsulating VisionModel in a non-Module class
        self.model = EncapsulatedVisionModel(cfg, device)
        self.last_layer = Last()
        self.last_layer.__class__.__name__ = 'last_layer'
        self.dummy_param = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, x):
        return self.last_layer(self.model.forward(x))

class VisionModel(nn.Module):
    def __init__(self, cfg, device):
        super(VisionModel, self).__init__()
        self.device = device
        self.wavelet_layers = WaveletLayers(cfg, self.device)
        out_channels = self.wavelet_layers.out_channels
        self.conv_layers = ConvolutionLayers(
            cfg, out_channels, self.device
        )
        self.last_layer = Last()
        self.last_layer.__class__.__name__ = 'last_layer'

    def forward(self, x):
        x = x.to(self.device)
        x = self.wavelet_layers(x)
        x = self.conv_layers(x)
        x = self.last_layer(x)
        
        return x


def AlexNet(pretrained=True):
    if pretrained:
        alexnet = models.alexnet(weights=AlexNet_Weights.DEFAULT)
    else:
        alexnet = models.alexnet()
    last_layer = alexnet.classifier[-1]
    last_layer.__class__.__name__ = "last_layer"

    return alexnet


def VGG16(pretrained=True):
    if pretrained:
        vggnet = models.vgg16(weights=VGG16_Weights.DEFAULT)
    else:
        vggnet = models.vgg16()

    last_layer = vggnet.classifier[-1]
    last_layer.__class__.__name__ = "last_layer"
    return vggnet


def ResNet50(pretrained=True):
    if pretrained:
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
        resnet = models.resnet50()

    last_layer = resnet.fc
    last_layer.__class__.__name__ = "last_layer"
    return resnet


def DenseNet121(pretrained=True):
    if pretrained:
        densenet = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
    else:
        densenet = models.densenet121()

    last_layer = densenet.classifier
    last_layer.__class__.__name__ = "last_layer"
    return densenet
