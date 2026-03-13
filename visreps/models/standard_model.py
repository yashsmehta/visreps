import torchvision.models as models
import torch
import torch.nn as nn
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

def ViTBase(pretrained_dataset="imagenet1k", num_classes=1000):
    """ViT-Base/16 with optional ImageNet pretraining."""
    if pretrained_dataset == "imagenet1k":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    elif pretrained_dataset == "none":
        model = models.vit_b_16(weights=None)
    else:
        raise ValueError(f"Invalid pretrained dataset: {pretrained_dataset}")
    
    # Replace classification head if num_classes differs
    if num_classes != 1000 and num_classes is not None:
        model.heads.head = torch.nn.Linear(768, num_classes)
        torch.nn.init.xavier_uniform_(model.heads.head.weight)
        torch.nn.init.zeros_(model.heads.head.bias)

    return model


class CLIPVisualExtractor(nn.Module):
    """CLIP ViT-L/14 visual encoder with per-block feature extraction.

    Runs the forward pass manually so that intermediate block outputs are
    captured in (N, L, D) format, compatible with the eval pipeline's
    flatten-then-SRP logic.
    """

    def __init__(self):
        super().__init__()
        import clip
        model, _ = clip.load("ViT-L/14", device="cpu")
        self.visual = model.visual.float()
        self.return_nodes = None  # set by configure_feature_extractor

    def forward(self, x):
        v = self.visual
        x = v.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        cls = v.class_embedding.to(x.dtype).unsqueeze(0).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + v.positional_embedding.to(x.dtype)
        x = v.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND (CLIP's internal format)

        features = {}
        for i, block in enumerate(v.transformer.resblocks):
            x = block(x)
            name = f"block{i + 1}"
            if self.return_nodes and name in self.return_nodes:
                # (L, N, D) -> (N, L, D) for pipeline compatibility
                features[self.return_nodes[name]] = x.permute(1, 0, 2).float()

        return features


def CLIP_ViT_L14(pretrained_dataset=None, num_classes=None):
    """Load CLIP ViT-L/14 visual encoder as a feature extractor."""
    return CLIPVisualExtractor()

