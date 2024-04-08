import torchvision.models as models

def AlexNet():
    alexnet = models.alexnet(pretrained=False)
    return alexnet
