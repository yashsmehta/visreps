import torchvision.models as models
import torch

def AlexNet(pretrained_dataset="imagenet1k", num_classes=1000):
    """AlexNet with optional ImageNet pretraining."""
    if pretrained_dataset == "imagenet1k":
        model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    elif pretrained_dataset == "none":
        model = models.alexnet(weights=None)
    else:
        raise ValueError(f"Invalid pretrained dataset: {pretrained_dataset}")
    
    # replace classifier if not using ImageNet (1000 classes)
    if num_classes != 1000 and num_classes is not None:
        model.classifier[-1] = torch.nn.Linear(4096, num_classes)
        torch.nn.init.xavier_uniform_(model.classifier[-1].weight)
        torch.nn.init.zeros_(model.classifier[-1].bias)
    
    return model

def VGG16(pretrained_dataset="imagenet1k", num_classes=200):
    """VGG16 with optional ImageNet pretraining."""
    if pretrained_dataset == "imagenet1k":
        model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
    elif pretrained_dataset == "none":
        model = models.vgg16(weights=None)
    else:
        raise ValueError(f"Invalid pretrained dataset: {pretrained_dataset}")
    
    # Replace classifier
    if num_classes is not None:
        model.classifier[-1] = torch.nn.Linear(4096, num_classes)
    
    # Initialize the classifier weights if not using pretrained model
    if pretrained_dataset == "none":
        torch.nn.init.xavier_uniform_(model.classifier[-1].weight)
        torch.nn.init.zeros_(model.classifier[-1].bias)
    
    return model

def ResNet18(pretrained_dataset="imagenet1k", num_classes=200):
    """ResNet18 with optional ImageNet pretraining."""
    if pretrained_dataset == "imagenet1k":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    elif pretrained_dataset == "none":
        model = models.resnet18(weights=None)
    else:
        raise ValueError(f"Invalid pretrained dataset: {pretrained_dataset}")
    
    # Replace fc layer
    if num_classes is not None:
        model.fc = torch.nn.Linear(512, num_classes)
    
    # Initialize the fc weights if not using pretrained model
    if pretrained_dataset == "none":
        torch.nn.init.xavier_uniform_(model.fc.weight)
        torch.nn.init.zeros_(model.fc.bias)
    
    return model

def ResNet50(pretrained_dataset="imagenet1k", num_classes=200):
    """ResNet50 with optional ImageNet pretraining."""
    if pretrained_dataset == "imagenet1k":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    elif pretrained_dataset == "none":
        model = models.resnet50(weights=None)
    else:
        raise ValueError(f"Invalid pretrained dataset: {pretrained_dataset}")
    
    # Replace fc layer
    if num_classes is not None:
        model.fc = torch.nn.Linear(2048, num_classes)
    
    # Initialize the fc weights if not using pretrained model
    if pretrained_dataset == "none" and num_classes is not None:
        torch.nn.init.xavier_uniform_(model.fc.weight)
        torch.nn.init.zeros_(model.fc.bias)
    
    return model
