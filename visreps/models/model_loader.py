import os
import torch
from visreps.models import standard_cnns
from visreps.models.custom_cnn import CustomCNN

def load_model(cfg):
    """Load a model from checkpoint or initialize a new model.
    
    Args:
        cfg: Configuration object containing:
            - model_name: Name of model or path to checkpoint
            - model_class: Type of model ('custom_cnn' or standard model name)
            - pretrained: Whether to use pretrained weights (for standard models)
            - num_classes: Number of output classes
            - custom: Custom model configuration (for CustomCNN)
        
    Returns:
        torch.nn.Module: Loaded model
        
    Raises:
        ValueError: If model_class is not recognized
    """
    if os.path.exists(cfg.model_name):
        # Load from checkpoint file
        checkpoint = torch.load(cfg.model_name)
        model = checkpoint['model']
        print(f"Loaded model from checkpoint: {cfg.model_name}")
        return model
    
    # Initialize new model
    if cfg.model_class == "custom_cnn":
        model = CustomCNN(
            num_classes=getattr(cfg, 'num_classes', 200),
            trainable_layers={
                'conv': cfg.custom.conv_trainable,
                'fc': cfg.custom.fc_trainable
            },
            nonlinearity=cfg.custom.nonlinearity,
            dropout=cfg.custom.dropout,
            batchnorm=cfg.custom.batchnorm,
            pooling_type=cfg.custom.pooling_type
        )
        print("Initialized custom CNN model")
    else:
        # Get standard model initialization function
        model_fn = getattr(standard_cnns, cfg.model_name, None)
        if model_fn is None:
            raise ValueError(f"Model class {cfg.model_name} not found")
        
        pretrained = getattr(cfg, 'pretrained', True)
        num_classes = getattr(cfg, 'num_classes', 200)
        model = model_fn(pretrained=pretrained, num_classes=num_classes)
        print(f"Initialized standard model: {cfg.model_name}")
    
    return model 