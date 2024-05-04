import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, AlexNet_Weights, VGG16_Weights, DenseNet121_Weights
from .conv_layers import ConvolutionLayers
from .wavelet_layers import WaveletLayers
from .last_layer import Last
import torch

class CustomAlexNet(nn.Module):
    def __init__(self, num_classes=200):
        super(CustomAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 1 * 1, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class EncapsulatedVisionModel:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.model = VisionModel(self.cfg, self.device).to(self.device)

    def forward(self, x):
        return self.model(x)


class VisionModelHack(nn.Module):
    def __init__(self, cfg, device):
        super(VisionModelHack, self).__init__()
        self.cfg = cfg
        self.device = device
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
    # last_conv_layer = alexnet.features[-1]
    # last_conv_layer.__class__.__name__ = "last_layer"

    return alexnet

def VGG16(pretrained=True):
    if pretrained:
        vggnet = models.vgg16(weights=VGG16_Weights.DEFAULT)
    else:
        vggnet = models.vgg16()

    last_conv_layer = vggnet.features[-1]
    last_conv_layer.__class__.__name__ = "last_layer"
    return vggnet


def ResNet50(pretrained=True):
    if pretrained:
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
        resnet = models.resnet50()

    last_conv_layer = resnet.layer4[-1]
    last_conv_layer.__class__.__name__ = "last_layer"
    return resnet

def DenseNet121(pretrained=True):

    if pretrained:
        densenet = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
    else:
        densenet = models.densenet121()

    for idx, layer in enumerate(densenet.features):
        if idx == len(densenet.features) - 1:
            layer.__class__.__name__ = "last_layer"

    return densenet
