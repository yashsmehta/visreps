"""
Shared utilities for coarse-grain benefits experiments.
"""

import os
import sys

# Get project root and add to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Ensure .env is loaded from project root
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

import torch
import numpy as np
from tqdm import tqdm
import torchvision.models as models

from visreps.models.utils import FeatureExtractor


# Default checkpoint directory and configurations
DEFAULT_CHECKPOINT_DIR = "/data/ymehta3/alexnet_pca"
DEFAULT_CHECKPOINT_MODEL = "checkpoint_epoch_20.pth"

# Output directory for results
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "experiments/coarse_grain_benefits/results")


def get_device():
    """Get the appropriate device (CUDA if available)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def load_pretrained_alexnet(device):
    """Load pretrained ImageNet AlexNet."""
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    return model.to(device).eval()


def load_checkpoint_model(checkpoint_path, device):
    """Load model from checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint['model'].to(device).eval()


def load_coarse_model(cfg_id, seed, checkpoint_dir=None, device=None):
    """Load a trained model by cfg_id and seed."""
    if device is None:
        device = get_device()
    if checkpoint_dir is None:
        checkpoint_dir = DEFAULT_CHECKPOINT_DIR

    seed_letter = chr(ord('a') + seed - 1)  # 1->a, 2->b, 3->c

    # 1000-way models are in /data/ymehta3/default/
    if cfg_id == 1000:
        checkpoint_path = f"/data/ymehta3/default/cfg{cfg_id}{seed_letter}/{DEFAULT_CHECKPOINT_MODEL}"
    else:
        checkpoint_path = os.path.join(checkpoint_dir, f"cfg{cfg_id}{seed_letter}", DEFAULT_CHECKPOINT_MODEL)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    return load_checkpoint_model(checkpoint_path, device)


def get_model_configs(cfg_ids=None, seeds=None, include_pretrained=False):
    """
    Return list of (cfg_id, seed) pairs to evaluate.

    Args:
        cfg_ids: List of cfg_ids to evaluate (default: [32, 64, 1000])
        seeds: List of seeds to evaluate (default: [1])
        include_pretrained: If True, also include torchvision pretrained as separate baseline

    Returns:
        List of (cfg_id, seed) tuples
        If include_pretrained=True, also includes ('pretrained', None) for torchvision baseline
    """
    if cfg_ids is None:
        cfg_ids = [32, 64, 1000]  # Now includes 1000-way from checkpoint by default
    if seeds is None:
        seeds = [1]

    configs = []
    for cfg_id in cfg_ids:
        for seed in seeds:
            configs.append((cfg_id, seed))

    # Optionally add torchvision pretrained as separate baseline
    if include_pretrained:
        configs.append(('pretrained', None))

    return configs


def load_model_by_config(cfg_id, seed, checkpoint_dir=None, device=None):
    """
    Load a model by configuration.

    Args:
        cfg_id: Configuration ID (32, 64, 1000) or 'pretrained' for torchvision
        seed: Training seed (1, 2, 3) or None for torchvision pretrained
        checkpoint_dir: Path to checkpoint directory
        device: Torch device

    Returns:
        Loaded model in eval mode
    """
    if device is None:
        device = get_device()

    if cfg_id == 'pretrained':
        # Load torchvision pretrained AlexNet (different training pipeline)
        return load_pretrained_alexnet(device)
    else:
        # Load from checkpoint (includes 1000-way trained with same pipeline)
        return load_coarse_model(cfg_id, seed, checkpoint_dir, device)


def get_feature_extractor(model, layers):
    """
    Wrap a model with a feature extractor for specified layers.

    Args:
        model: PyTorch model
        layers: List of layer names (e.g., ['fc2']) or single layer name

    Returns:
        FeatureExtractor that returns features from specified layers
    """
    if isinstance(layers, str):
        layers = [layers]

    return_nodes = {layer: layer for layer in layers}
    extractor = FeatureExtractor(model, return_nodes=return_nodes)
    return extractor.eval()


def extract_features(model, loader, layer='fc2', device=None, show_progress=True):
    """
    Extract frozen features from a model for a given dataloader.

    Args:
        model: PyTorch model (will be wrapped with FeatureExtractor)
        loader: DataLoader
        layer: Layer name to extract (default: 'fc2')
        device: Torch device (default: auto-detect)
        show_progress: Whether to show progress bar

    Returns:
        features: numpy array of shape (n_samples, n_features)
        labels: numpy array of shape (n_samples,)
    """
    if device is None:
        device = get_device()

    extractor = get_feature_extractor(model, layer)
    extractor.to(device).eval()

    features_list = []
    labels_list = []

    iterator = tqdm(loader, desc=f"Extracting {layer}", leave=False) if show_progress else loader

    with torch.no_grad():
        for images, labels in iterator:
            feats = extractor(images.to(device))
            # Flatten if needed (conv layers have spatial dimensions)
            feat = feats[layer]
            if feat.dim() > 2:
                feat = feat.view(feat.size(0), -1)
            features_list.append(feat.cpu().numpy())
            labels_list.append(labels.numpy())

    return np.vstack(features_list), np.concatenate(labels_list)


def extract_features_batch(model, images, layer='fc2', device=None):
    """
    Extract features for a batch of images.

    Args:
        model: PyTorch model
        images: Tensor of images (B, C, H, W)
        layer: Layer name to extract
        device: Torch device

    Returns:
        features: numpy array of shape (B, n_features)
    """
    if device is None:
        device = get_device()

    extractor = get_feature_extractor(model, layer)
    extractor.to(device).eval()

    with torch.no_grad():
        feats = extractor(images.to(device))
        feat = feats[layer]
        if feat.dim() > 2:
            feat = feat.view(feat.size(0), -1)
        return feat.cpu().numpy()


def get_config_name(cfg_id, seed):
    """Get a human-readable name for a configuration."""
    if cfg_id == 'pretrained':
        return "Torchvision Pretrained"
    else:
        seed_letter = chr(ord('a') + seed - 1) if seed else ''
        return f"{cfg_id}-way{seed_letter}"
