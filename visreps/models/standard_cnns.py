import torchvision.models as models
from torchvision.models import ResNet50_Weights, AlexNet_Weights, VGG16_Weights, DenseNet121_Weights


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