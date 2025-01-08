import json
import os
import torch
import torchvision
from omegaconf import OmegaConf
import torch.nn as nn
from typing import Dict

from visreps.models import standard_cnn
from visreps.models.custom_cnn import CustomCNN

class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, return_nodes: Dict[str, str] = None):
        super().__init__()
        self.model = model
        # If return_nodes is a list, convert to dict mapping name to itself
        if isinstance(return_nodes, list):
            return_nodes = {node: node for node in return_nodes}
        self.return_nodes = return_nodes
        self.features = {}
        self.handles = []  # Initialize handles list
        self.layer_mapping = self._create_layer_mapping()
        self._attach_hooks()
        
    def _create_layer_mapping(self):
        """Create mapping from semantic names (conv1, fc1) to actual layer paths"""
        mapping = {}
        conv_count = 1
        fc_count = 1
        
        # Helper function to check if module is a conv or fc layer
        def is_conv(module):
            return isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d))
        def is_fc(module):
            return isinstance(module, nn.Linear)
        def is_main_conv(name, module):
            """Check if this is a main conv layer (not a downsample/shortcut conv)"""
            return is_conv(module) and 'downsample' not in name
        
        # Track seen modules to avoid duplicates
        seen_modules = set()
        
        # Handle different model architectures
        if isinstance(self.model, torchvision.models.resnet.ResNet):  # Fixed ResNet check
            # ResNet specific mapping - maintain proper layer order
            # First conv layer is special in ResNet
            if hasattr(self.model, 'conv1'):
                mapping[f'conv{conv_count}'] = 'conv1'
                conv_count += 1
                seen_modules.add(id(self.model.conv1))
            
            # Then handle layer blocks in order
            for block_name in ['layer1', 'layer2', 'layer3', 'layer4']:
                if hasattr(self.model, block_name):
                    block = getattr(self.model, block_name)
                    for sub_block_idx, sub_block in enumerate(block):
                        # Get all convs in this sub_block except downsampling
                        for name, module in sub_block.named_modules():
                            if is_main_conv(name, module) and id(module) not in seen_modules:
                                full_name = f'{block_name}.{sub_block_idx}.{name}'
                                mapping[f'conv{conv_count}'] = full_name
                                conv_count += 1
                                seen_modules.add(id(module))
            
            # Handle final fc layer
            if hasattr(self.model, 'fc'):
                mapping['fc1'] = 'fc'
                seen_modules.add(id(self.model.fc))
                
        elif hasattr(self.model, 'features') and hasattr(self.model, 'classifier'):
            # AlexNet/VGG style
            for name, module in self.model.features.named_modules():
                if is_conv(module) and id(module) not in seen_modules:
                    mapping[f'conv{conv_count}'] = f'features.{name}'
                    conv_count += 1
                    seen_modules.add(id(module))
                    
            for name, module in self.model.classifier.named_modules():
                if is_fc(module) and id(module) not in seen_modules:
                    mapping[f'fc{fc_count}'] = f'classifier.{name}'
                    fc_count += 1
                    seen_modules.add(id(module))
                    
        # Fallback: search all modules if no specific pattern matched
        if not mapping:
            for name, module in self.model.named_modules():
                if id(module) in seen_modules:
                    continue
                    
                if is_main_conv(name, module):
                    mapping[f'conv{conv_count}'] = name
                    conv_count += 1
                    seen_modules.add(id(module))
                elif is_fc(module):
                    mapping[f'fc{fc_count}'] = name
                    fc_count += 1
                    seen_modules.add(id(module))
        
        for semantic_name, path in sorted(mapping.items()):
            module = dict(self.model.named_modules())[path]
        
        return mapping
        
    def _attach_hooks(self):
        # Convert semantic names to actual layer paths
        actual_nodes = {}
        for semantic_name in self.return_nodes:
            if semantic_name in self.layer_mapping:
                actual_path = self.layer_mapping[semantic_name]
                actual_nodes[actual_path] = self.return_nodes[semantic_name]
            else:
                print(f"Warning: {semantic_name} not found in model")
        
        # Attach hooks using actual paths
        for name, module in self.model.named_modules():
            if name in actual_nodes:
                def get_hook(name):
                    def hook(module, input, output):
                        self.features[actual_nodes[name]] = output
                    return hook
                
                handle = module.register_forward_hook(get_hook(name))
                self.handles.append(handle)
    
    def forward(self, x):
        self.features.clear()
        self.model(x)
        return self.features
    
    def __del__(self):
        for handle in self.handles:
            handle.remove()

def configure_feature_extractor(cfg, model):
    return_nodes = OmegaConf.to_container(cfg.get("return_nodes", {}), resolve=True)
    if not return_nodes:
        raise ValueError("return_nodes must be specified in config")
    return_nodes = {node: node for node in return_nodes} if isinstance(return_nodes, list) else return_nodes
    # Use our custom feature extractor
    model.eval()  # Ensure consistent behavior
    print(f"Extracting features from layers: {list(return_nodes.keys())}")
    return FeatureExtractor(model, return_nodes)


def get_activations(model, dataloader, device):
    """
    Extract activations from a model for a given dataloader.
    """
    model.eval()
    activations_dict = {}
    all_keys = []
    with torch.no_grad():
        for images, keys in dataloader:
            inputs = images.to(device)
            outputs = model(inputs)
            all_keys.extend(keys)
            for node_name, output in outputs.items():
                output = output.cpu()
                activations_dict.setdefault(node_name, []).append(output)
    for node_name in activations_dict:
        activations_dict[node_name] = torch.cat(activations_dict[node_name], dim=0)
    return activations_dict, all_keys


def merge_checkpoint_config(cfg, checkpoint):
    """
    Update cfg in place with config from checkpoint, giving priority to current cfg.
    Only adds keys from checkpoint that don't exist in current cfg.
    
    Args:
        cfg (OmegaConf): Current configuration
        checkpoint (dict): Loaded checkpoint containing 'config' and optionally 'epoch'
    """
    if 'config' in checkpoint:
        checkpoint_cfg = OmegaConf.create(checkpoint['config'])
        # Update only missing keys from checkpoint config
        for key in checkpoint_cfg:
            if key not in cfg:
                cfg[key] = checkpoint_cfg[key]
    if 'epoch' in checkpoint:
        cfg.epoch = checkpoint['epoch']


def load_model(cfg, device):
    """
    Load a model from checkpoint or initialize a new model.

    Args:
        cfg: Configuration with keys depending on mode:
            - For training:
                model_class ('custom_cnn'/'standard_cnn'),
                model_name,
                pretrained,
                custom (dict with additional args for CustomCNN)
            - For evaluation:
                load_model_from ('torchvision'/'checkpoint'),
                model_name,
                checkpoint_path
        device: Torch device (e.g., 'cpu' or 'cuda').

    Returns:
        torch.nn.Module: The loaded or newly-initialized model.
    """
    # If loading from checkpoint, merge configs and return
    if getattr(cfg, 'load_model_from', None) == 'checkpoint':
        checkpoint_path = cfg.checkpoint_path
        checkpoint = torch.load(checkpoint_path, map_location=device)
        print(f"Loaded model from checkpoint: {checkpoint_path}")
        merge_checkpoint_config(cfg, checkpoint)
        return checkpoint['model'].to(device)

    # Otherwise, determine model class and name
    model_class = getattr(cfg, 'model_class', 'standard_cnn')
    model_name = getattr(cfg, 'model_name', 'AlexNet')

    # Initialize custom CNN
    if model_class == 'custom_cnn':
        custom_cfg = getattr(cfg, 'arch', {})
        model_params = {
            'num_classes': getattr(cfg, 'num_classes', 200),
            'trainable_layers': {
                'conv': getattr(custom_cfg, 'conv_trainable', '11111'),
                'fc': getattr(custom_cfg, 'fc_trainable', '111')
            },
            'nonlinearity': getattr(custom_cfg, 'nonlinearity', 'relu'),
            'dropout': getattr(custom_cfg, 'dropout', True),
            'batchnorm': getattr(custom_cfg, 'batchnorm', True),
            'pooling_type': getattr(custom_cfg, 'pooling_type', 'max')
        }
        model = CustomCNN(**model_params)
        
    else:
        # Initialize standard CNN
        model_fn = getattr(standard_cnn, model_name, None)
        if model_fn is None:
            raise ValueError(f"Model '{model_name}' not found in standard_cnn.")
        pretrained_dataset = getattr(cfg, 'pretrained_dataset', "none")
        num_classes = getattr(cfg, 'num_classes', 200)
        
        # Load model - classifier layer is already replaced in model_fn
        model = model_fn(pretrained_dataset, num_classes)
        
        # Initialize the classifier weights if not using pretrained model
        if pretrained_dataset == "none":
            if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
                # For models like AlexNet, VGG
                nn.init.xavier_uniform_(model.classifier[-1].weight)
                nn.init.zeros_(model.classifier[-1].bias)
            elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
                # For models like DenseNet
                nn.init.xavier_uniform_(model.classifier.weight)
                nn.init.zeros_(model.classifier.bias)
            elif hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
                # For models like ResNet
                nn.init.xavier_uniform_(model.fc.weight)
                nn.init.zeros_(model.fc.bias)
            else:
                raise ValueError(f"Unknown classifier structure for model {model_name}")

    return model.to(device)


def make_checkpoint_dir(folder, parent_dir="model_checkpoints"):
    """
    Create a new subdirectory for a training checkpoint within a specified parent directory.

    This function constructs a directory path using the provided `folder` name under the `parent_dir`.
    It ensures that the directory exists and then creates a new subdirectory within it to store the
    checkpoint. The subdirectory is named by incrementing the count of existing directories.

    Args:
        folder (str): The name of the main folder under which the checkpoint directory will be created.
        parent_dir (str): The parent directory where the `folder` will be located. Defaults to 'model_checkpoints'.

    Returns:
        str: The path to the newly created checkpoint subdirectory.
    """
    checkpoint_dir = os.path.join(parent_dir, folder)
    os.makedirs(checkpoint_dir, exist_ok=True)
    ith_folder = len(os.listdir(checkpoint_dir)) + 1
    checkpoint_subdir = os.path.join(checkpoint_dir, "cfg" + str(ith_folder))
    os.makedirs(checkpoint_subdir, exist_ok=True)
    return checkpoint_subdir


def setup_checkpoint_dir(cfg, model):
    """
    Create the checkpoint directory (if needed), store config info, and return
    the directory path and the extended config dict.
    """
    checkpoint_dir = make_checkpoint_dir(cfg.exp_name)
    cfg_dict = {
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        **OmegaConf.to_container(cfg, resolve=True),
    }
    # Save the config.json
    with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=2)
    
    return checkpoint_dir, cfg_dict


def save_checkpoint(checkpoint_dir, epoch, model, optimizer, metrics, cfg_dict):
    """Save model and optimizer states along with metrics and config at a checkpoint."""
    torch.save(
        {
            "epoch": epoch,
            "model": model,  # Save full model
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": cfg_dict,
        },
        os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    )
