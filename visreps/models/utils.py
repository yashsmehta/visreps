from collections import defaultdict
import json
import os
import sys
import torch
import torchvision
from omegaconf import OmegaConf
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np
import joblib
from tqdm import tqdm
import psutil                     # Added import
import resource                   # Added import (optional, for peak process memory)

from visreps.models import standard_model
from visreps.models import custom_model
from visreps.models.custom_model import CustomCNN, TinyCustomCNN

# Backward compat: old checkpoints reference 'visreps.models.custom_cnn'
sys.modules['visreps.models.custom_cnn'] = custom_model
from visreps.utils import rprint
from visreps.utils import get_seed_letter
from visreps.analysis.sparse_random_projection import get_srp_transformer

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
        
        elif isinstance(self.model, torchvision.models.VisionTransformer):
            # ViT specific mapping
            # Patch embedding conv
            if hasattr(self.model, 'conv_proj'):
                mapping['patch_embed'] = 'conv_proj'
                seen_modules.add(id(self.model.conv_proj))
            
            # Transformer encoder blocks (block1, block2, ..., block12 for ViT-Base)
            # torchvision names them encoder.layers.encoder_layer_{i}
            if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layers'):
                for i, block in enumerate(self.model.encoder.layers):
                    mapping[f'block{i+1}'] = f'encoder.layers.encoder_layer_{i}'
                    seen_modules.add(id(block))
            
            # Classification head
            if hasattr(self.model, 'heads') and hasattr(self.model.heads, 'head'):
                mapping['head'] = 'heads.head'
                seen_modules.add(id(self.model.heads.head))
                
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


def get_activations(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    apply_srp: bool = False,
) -> Tuple[Dict[str, torch.Tensor], List]:
    """
    Collect layer activations for every sample in `dataloader`.
    If `apply_srp`, each batch is projected (Sparse Random Projection) to
    k = 4096 per layer, capped by its native dimensionality.
    """
    model.eval()
    activations: Dict[str, List[torch.Tensor]] = defaultdict(list)
    ids: List = []
    srp = {}

    # ---------- SRP initialisation ----------
    if apply_srp:
        try:
            probe_imgs, _ = next(iter(dataloader))                # single probe batch
        except StopIteration:
            return {}, []

        with torch.no_grad():
            probe_out = model(probe_imgs.to(device))

        k_fixed, density, seed, cache_dir = 4096, None, None, "model_checkpoints/srp_cache"
        num_layers = len(probe_out)
        for name, out in probe_out.items():
            D = out.view(out.size(0), -1).size(1)
            transformer = get_srp_transformer(
                D=D,
                k=min(k_fixed, D),
                density=density,
                seed=seed,
                cache_dir=cache_dir,
            )
            # Convert sklearn sparse matrix to PyTorch sparse tensor on GPU
            sparse_matrix = transformer.components_  # shape (k, D)
            coo = sparse_matrix.tocoo()
            indices = torch.from_numpy(np.vstack([coo.row, coo.col])).long()
            values = torch.from_numpy(coo.data).float()
            proj_matrix = torch.sparse_coo_tensor(indices, values, coo.shape).to(device)
            srp[name] = proj_matrix

        rprint(f"✓ Loaded SRP transformers for {num_layers} layers", style="success")

    # ---------- main loop ----------
    FP16_MAX = 65504.0  # float16 max value
    with torch.no_grad():
        for imgs, keys in tqdm(dataloader, desc="Extracting model activations"):
            ids.extend(keys)
            feats = model(imgs.to(device))
            for name, out in feats.items():
                proj_matrix = srp.get(name)
                if apply_srp and proj_matrix is not None:
                    flat = out.view(out.size(0), -1).float()  # (batch, D) on GPU
                    out = torch.sparse.mm(proj_matrix, flat.t()).t()  # (batch, k)
                # Clamp to float16 range before half() to avoid overflow → inf
                activations[name].append(out.clamp(-FP16_MAX, FP16_MAX).cpu().half())

    return {n: torch.cat(b, 0) for n, b in activations.items()}, ids


def load_model(cfg, device, num_classes=None):
    """
    Load a model from checkpoint or initialize a new model.

    Args:
        cfg: Configuration with keys depending on mode:
            - For training:
                model_class ('custom_model'/'standard_model'),
                model_name,
                pretrained,
                custom (dict with additional args for CustomCNN)
            - For evaluation:
                load_model_from ('torchvision'/'checkpoint'),
                model_name,
                checkpoint_path
        device: Torch device (e.g., 'cpu' or 'cuda').
        num_classes: Number of output classes.

    Returns:
        torch.nn.Module: The loaded or newly-initialized model.
    """
    # If loading from checkpoint, merge configs and return
    if getattr(cfg, 'load_model_from', None) == 'checkpoint':
        if num_classes is not None:
            rprint("WARNING: num_classes is ignored when loading from checkpoint", style="warning")
        seed_letter = get_seed_letter(cfg.seed)
        checkpoint_path = f"{cfg.checkpoint_dir}/cfg{cfg.cfg_id}{seed_letter}/{cfg.checkpoint_model}"
        # Explicitly set weights_only=False to allow loading pickled code
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        print(f"Loaded model from checkpoint: {checkpoint_path}")
        return checkpoint['model'].to(device)

    # Otherwise, determine model class and name
    model_class = getattr(cfg, 'model_class', 'standard_model')
    model_name = getattr(cfg, 'model_name', 'AlexNet')

    # Initialize custom CNN
    if model_class == 'custom_model':
        custom_cfg = getattr(cfg, 'arch', {})
        model_params = {
            'num_classes': num_classes,
            'trainable_layers': {
                'conv': getattr(custom_cfg, 'conv_trainable', '11111'),
                'fc': getattr(custom_cfg, 'fc_trainable', '111')
            },
            'dropout': getattr(custom_cfg, 'dropout', 0.5),
            'pooling_type': getattr(custom_cfg, 'pooling_type', 'max')
        }
        if 'tiny' in model_name.lower():
            model = TinyCustomCNN(**model_params)
        else:
            model = CustomCNN(**model_params)
        
    else:
        # Initialize standard CNN
        model_fn = getattr(standard_model, model_name, None)
        if model_fn is None:
            raise ValueError(f"Model '{model_name}' not found in standard_model.")
        pretrained_dataset = getattr(cfg, 'pretrained_dataset', "none")
        
        # Load model - classifier layer is already replaced in model_fn
        model = model_fn(pretrained_dataset, num_classes)

    return model.to(device)


def setup_checkpoint_dir(cfg, model):
    """
    Create the checkpoint directory with proper naming scheme and store config info.

    Naming scheme:
    - Both PCA and non-PCA: cfg{num_classes}{seed_letter}
    """
    # Validate seed and get letter
    seed_letter = get_seed_letter(cfg.seed)

    # Determine cfg number based on actual number of classes
    if getattr(cfg, 'pca_labels', False):
        cfg_num = cfg.pca_n_classes
    else:
        # Tiny-ImageNet has 200 classes, everything else (ImageNet variants) has 1000
        cfg_num = 200 if cfg.get('dataset') == 'tiny-imagenet' else 1000

    # Create checkpoint path
    subdir_name = f"cfg{cfg_num}{seed_letter}"
    checkpoint_path = os.path.join("model_checkpoints", cfg.checkpoint_dir, subdir_name)
    os.makedirs(checkpoint_path, exist_ok=True)

    # Prepare and save config
    cfg_dict = {
        "total_params": sum(p.numel() for p in model.parameters()),
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        **OmegaConf.to_container(cfg, resolve=True),
    }

    with open(os.path.join(checkpoint_path, "config.json"), "w") as f:
        json.dump(cfg_dict, f, indent=2)

    return checkpoint_path, cfg_dict


def save_checkpoint(checkpoint_dir, epoch, model, optimizer, metrics, cfg_dict):
    """Save model and optimizer states along with metrics and config at a checkpoint."""
    torch.save(
        {
            "epoch": epoch,
            "model": model,
            # "optimizer_state_dict": optimizer.state_dict(), don't save optimizer state dict
            "metrics": metrics,
            "config": cfg_dict,
        },
        os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
    )
