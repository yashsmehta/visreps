import os
import torch
from visreps.models import standard_cnn
from visreps.models.custom_cnn import CustomCNN

def load_model(cfg):
    """Load a model from checkpoint or initialize a new model.
    
    Args:
        cfg: Configuration object containing either:
            Training config:
                - model_class: Type of model ('custom_cnn' or 'standard_cnn')
                - model_name: Name of model
                - pretrained: Whether to use pretrained weights
                - custom: Custom model configuration (for CustomCNN)
            Evaluation config:
                - load_model_from: Source of model ('torchvision' or 'checkpoint')
                - model_name: Name of torchvision model
                - checkpoint_path: Path to model checkpoint
        
    Returns:
        torch.nn.Module: Loaded model
    """
    # Handle evaluation config
    load_from = getattr(cfg, 'load_model_from', None)
    if load_from == 'checkpoint':
        checkpoint = torch.load(cfg.checkpoint_path)
        model = checkpoint['model']
        print(f"Loaded model from checkpoint: {cfg.checkpoint_path}")
        return model
    
    # Get model name, defaulting to AlexNet for torchvision
    model_name = getattr(cfg, 'model_name', 'AlexNet')
    
    # Handle training config or torchvision loading
    model_class = getattr(cfg, 'model_class', 'standard_cnn')
    
    if model_class == "custom_cnn":
        # Get custom model config with defaults
        custom_cfg = getattr(cfg, 'custom', {})
        model = CustomCNN(
            num_classes=getattr(cfg, 'num_classes', 200),
            trainable_layers={
                'conv': getattr(custom_cfg, 'conv_trainable', '11111'),
                'fc': getattr(custom_cfg, 'fc_trainable', '111')
            },
            nonlinearity=getattr(custom_cfg, 'nonlinearity', 'relu'),
            dropout=getattr(custom_cfg, 'dropout', True),
            batchnorm=getattr(custom_cfg, 'batchnorm', True),
            pooling_type=getattr(custom_cfg, 'pooling_type', 'max')
        )
        print("Initialized custom CNN model")
    else:
        # Get standard model initialization function
        model_fn = getattr(standard_cnn, model_name, None)
        if model_fn is None:
            raise ValueError(f"Model {model_name} not found in standard_cnn")
        
        pretrained = getattr(cfg, 'pretrained', True)
        num_classes = getattr(cfg, 'num_classes', 200)
        model = model_fn(pretrained=pretrained, num_classes=num_classes)
        print(f"Initialized standard model: {model_name}")
    
    return model 