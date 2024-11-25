import torchvision.models as models
from torchvision.models import ResNet50_Weights, AlexNet_Weights, VGG16_Weights, DenseNet121_Weights
import torch


def AlexNet(pretrained=True, num_classes=200):
    if pretrained:
        alexnet = models.alexnet(weights=AlexNet_Weights.DEFAULT)
    else:
        alexnet = models.alexnet()

    alexnet.classifier[-1] = torch.nn.Linear(4096, num_classes)
    return alexnet

def VGG16(pretrained=True, num_classes=200):
    if pretrained:
        vggnet = models.vgg16(weights=VGG16_Weights.DEFAULT)
    else:
        vggnet = models.vgg16()

    vggnet.classifier[-1] = torch.nn.Linear(4096, num_classes)
    return vggnet


def ResNet50(pretrained=True, num_classes=200):
    if pretrained:
        resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
        resnet = models.resnet50()

    resnet.fc = torch.nn.Linear(2048, num_classes)
    return resnet

def DenseNet121(pretrained=True, num_classes=200):
    if pretrained:
        densenet = models.densenet121(weights=DenseNet121_Weights.DEFAULT)
    else:
        densenet = models.densenet121()

    densenet.classifier = torch.nn.Linear(1024, num_classes)
    return densenet
