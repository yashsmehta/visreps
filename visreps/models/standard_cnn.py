import torchvision.models as models
import torch
from .ecnet import ECTiedNet as _ECTiedNet

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

def ResNet18(pretrained_dataset="imagenet1k", num_classes=1000):
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

def ECTiedNet(pretrained_dataset="none", num_classes=1000, C=192, expansion=1, N=4,
              dilations=None, mid_blurpool=True, dropout=0.3):
    """ECTiedNet: Weight-tied Expansion-Contraction CNN.

    Args:
        pretrained_dataset: Only "none" is supported (no pretraining available)
        num_classes: Number of output classes
        C: Base channel dimension
        expansion: Expansion ratio in ECBlock
        N: Number of times to apply the ECBlock
        dilations: Dilation schedule (list of ints)
        mid_blurpool: Whether to apply BlurPool downsampling at midpoint
        dropout: Dropout probability in classifier head
    """
    if pretrained_dataset not in ["none", None]:
        raise ValueError(f"ECTiedNet does not support pretrained weights. Use pretrained_dataset='none'")

    model = _ECTiedNet(
        num_classes=num_classes,
        C=C,
        expansion=expansion,
        N=N,
        dilations=dilations,
        mid_blurpool=mid_blurpool,
        dropout=dropout
    )

    return model
