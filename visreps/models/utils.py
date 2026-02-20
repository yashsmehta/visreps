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
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, MofNCompleteColumn
import psutil                     # Added import
import resource                   # Added import (optional, for peak process memory)

from visreps.models import standard_model
from visreps.models import custom_model
from visreps.models.custom_model import CustomCNN, TinyCustomCNN

# Backward compat: old checkpoints reference 'visreps.models.custom_cnn'
sys.modules['visreps.models.custom_cnn'] = custom_model
from visreps.utils import console, rprint
from visreps.utils import get_seed_letter
from visreps.analysis.sparse_random_projection import get_srp_transformer

class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, return_nodes: Dict[str, str] = None,
                 post_relu: bool = True, extract_pre_and_post: bool = True):
        super().__init__()
        self.model = model
        # If return_nodes is a list, convert to dict mapping name to itself
        if isinstance(return_nodes, list):
            return_nodes = {node: node for node in return_nodes}
        self.return_nodes = return_nodes
        self.post_relu = post_relu
        self.extract_pre_and_post = extract_pre_and_post
        self.features = {}
        self.handles = []  # Initialize handles list

        base_mapping = self._create_layer_mapping()

        if self.extract_pre_and_post:
            post_mapping = self._remap_to_post_relu(base_mapping)
            self.layer_mapping, self.return_nodes = self._build_pre_post_mapping(
                base_mapping, post_mapping
            )
        elif self.post_relu:
            self.layer_mapping = self._remap_to_post_relu(base_mapping)
        else:
            self.layer_mapping = base_mapping

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

    def _remap_to_post_relu(self, mapping):
        """Remap layer paths from Conv2d/Linear to their downstream activation fn.

        For Sequential containers (AlexNet/VGG/CustomCNN), searches forward from
        each mapped module to find the next ReLU/GELU/LeakyReLU in the sequence.
        This gives post-BatchNorm, post-ReLU activations — the actual output that
        downstream layers (and, by analogy, downstream brain areas) receive.
        """
        relu_mapping = {}
        requested = set(self.return_nodes or [])
        for semantic_name, module_path in mapping.items():
            if requested and semantic_name not in requested:
                relu_mapping[semantic_name] = module_path
                continue
            parts = module_path.split('.')
            if len(parts) == 2:
                container_name, idx_str = parts
                container = getattr(self.model, container_name, None)
                if container is not None and isinstance(container, nn.Sequential):
                    try:
                        module_idx = int(idx_str)
                    except ValueError:
                        relu_mapping[semantic_name] = module_path
                        continue
                    found = False
                    for i in range(module_idx + 1, len(container)):
                        if isinstance(container[i], (nn.ReLU, nn.GELU, nn.LeakyReLU)):
                            relu_mapping[semantic_name] = f'{container_name}.{i}'
                            found = True
                            break
                        # Stop if we hit another Conv/Linear (no activation for this layer)
                        if isinstance(container[i], (nn.Conv2d, nn.Conv1d, nn.Conv3d, nn.Linear)):
                            break
                    if not found:
                        print(f"Warning: No activation fn found after {semantic_name} "
                              f"({module_path}), keeping pre-activation")
                        relu_mapping[semantic_name] = module_path
                else:
                    relu_mapping[semantic_name] = module_path
            else:
                # Non-Sequential paths (e.g. ResNet 'layer1.0.conv1') — keep original
                relu_mapping[semantic_name] = module_path
        return relu_mapping

    def _build_pre_post_mapping(self, base_mapping, post_mapping):
        """Build expanded mapping with _pre and _post entries for each layer.

        _pre  = raw Conv2d/Linear output (before BatchNorm and ReLU)
        _post = post-BatchNorm, post-ReLU output

        Layers where no activation function was found (base == post path)
        are kept as a single entry with no suffix.
        """
        combined_mapping = {}
        expanded_return_nodes = {}

        for semantic_name, output_name in (self.return_nodes or {}).items():
            base_path = base_mapping.get(semantic_name)
            post_path = post_mapping.get(semantic_name)

            if base_path is None:
                print(f"Warning: {semantic_name} not found in base mapping")
                continue

            if post_path is not None and base_path != post_path:
                pre_name = f"{semantic_name}_pre"
                post_name = f"{semantic_name}_post"
                combined_mapping[pre_name] = base_path
                combined_mapping[post_name] = post_path
                expanded_return_nodes[pre_name] = pre_name
                expanded_return_nodes[post_name] = post_name
            else:
                # No activation found downstream — keep single entry
                combined_mapping[semantic_name] = base_path
                expanded_return_nodes[semantic_name] = output_name

        return combined_mapping, expanded_return_nodes

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

def configure_feature_extractor(cfg, model, verbose=False):
    return_nodes = OmegaConf.to_container(cfg.get("return_nodes", {}), resolve=True)
    if not return_nodes:
        raise ValueError("return_nodes must be specified in config")
    return_nodes = {node: node for node in return_nodes} if isinstance(return_nodes, list) else return_nodes
    extract_pre_and_post = cfg.get("extract_pre_and_post", True)
    model.eval()
    extractor = FeatureExtractor(model, return_nodes,
                                 extract_pre_and_post=extract_pre_and_post)
    n_points = len(extractor.return_nodes)
    n_layers = len(return_nodes)
    suffix = f" ({n_layers} layers × pre/post)" if extract_pre_and_post else ""
    rprint(f"  ✓ {n_points} extraction points{suffix}", style="success")
    if verbose:
        rprint(f"    Layers: {list(return_nodes.keys())}", style="info")
        rprint(f"    Points: {list(extractor.return_nodes.keys())}", style="info")
    return extractor


def get_activations(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], List]:
    """
    Collect layer activations for every sample in `dataloader`.
    Always applies Sparse Random Projection (k=4096 per layer, capped by
    native dimensionality) to keep memory bounded.
    """
    model.eval()
    activations: Dict[str, List[torch.Tensor]] = defaultdict(list)
    ids: List = []
    srp = {}

    # ---------- SRP initialisation ----------
    try:
        probe_imgs, _ = next(iter(dataloader))
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

    rprint(f"  ✓ SRP transformers for {num_layers} layers (k={k_fixed})", style="success")

    # ---------- main loop ----------
    with torch.no_grad(), Progress(
        TextColumn("  Extracting activations"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("batches"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("extract", total=len(dataloader))
        for imgs, keys in dataloader:
            ids.extend(keys)
            feats = model(imgs.to(device))
            for name, out in feats.items():
                proj_matrix = srp.get(name)
                if proj_matrix is not None:
                    flat = out.view(out.size(0), -1).float()  # (batch, D) on GPU
                    out = torch.sparse.mm(proj_matrix, flat.t()).t()  # (batch, k)
                activations[name].append(out.cpu().float())
            progress.advance(task)

    return {n: torch.cat(b, 0) for n, b in activations.items()}, ids


def extract_single_layer(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    layer_name: str,
    stimulus_ids: List[str] = None,
) -> Tuple[torch.Tensor, List]:
    """Re-extract one layer's full-resolution activations without SRP.

    Used after layer selection to get exact (un-projected) activations for
    the best layer only, keeping memory bounded since we store just one layer.

    Args:
        model: FeatureExtractor (same as used for SRP extraction).
        dataloader: Same dataloader (covers all stimuli, shuffle=False).
        device: GPU device.
        layer_name: Layer to extract, e.g. "conv5_post".
        stimulus_ids: If provided, only keep activations for these IDs.

    Returns:
        (acts_tensor, ordered_ids): float32 on CPU. If stimulus_ids is
        provided, rows are ordered to match stimulus_ids.
    """
    model.eval()
    all_acts = []
    all_ids = []

    with torch.no_grad(), Progress(
        TextColumn(f"  Re-extracting {layer_name}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("batches"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("re-extract", total=len(dataloader))
        for imgs, keys in dataloader:
            all_ids.extend(keys)
            feats = model(imgs.to(device))
            out = feats[layer_name]
            flat = out.view(out.size(0), -1)
            all_acts.append(flat.cpu().float())
            progress.advance(task)

    acts = torch.cat(all_acts, 0)

    if stimulus_ids is not None:
        # Reorder to match requested stimulus_ids
        id_to_idx = {str(k): i for i, k in enumerate(all_ids)}
        keep_idx = [id_to_idx[str(sid)] for sid in stimulus_ids if str(sid) in id_to_idx]
        acts = acts[keep_idx]
        all_ids = [all_ids[i] for i in keep_idx]

    rprint(f"  ✓ Re-extracted {layer_name}: {acts.shape} (exact, no SRP)", style="success")
    return acts, all_ids


def load_model(cfg, device, num_classes=None, verbose=False):
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
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        rprint(f"  ✓ Loaded checkpoint (cfg{cfg.cfg_id}{seed_letter})", style="success")
        if verbose:
            rprint(f"    Path: {checkpoint_path}", style="info")
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
