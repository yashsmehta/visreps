"""
Shared utilities for representation analysis experiments.
"""

import os
import sys

# Get project root and add to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

# Ensure .env is loaded from project root (before importing visreps modules)
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from visreps.models.utils import FeatureExtractor
from visreps.dataloaders.obj_cls import get_obj_cls_loader
import torchvision.models as models


# =============================================================================
# Configuration (shared across all analysis scripts)
# =============================================================================
DATASET = "imagenet-mini-50"
LAYER = "fc2"
CHECKPOINT_32WAY = "/data/ymehta3/alexnet_pca/cfg32a/checkpoint_epoch_20.pth"

# Resolve relative paths from project root
PCA_LABELS_PATH = os.path.join(PROJECT_ROOT, "pca_labels/pca_labels_alexnet/n_classes_32.csv")
SEMANTIC_LABELS_PATH = os.path.join(PROJECT_ROOT, "experiments/wordnet/semantic_categories.csv")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "experiments/representation_analysis")

# All layers for multi-layer analysis (conv1-5, fc1-2)
ALL_LAYERS = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc1', 'fc2']

MODEL_NAMES = ['Pretrained (1000-way)', '32-way Trained']
COLORS_32CLASS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3',
                  '#ff7f00', '#ffff33', '#a65628', '#f781bf',
                  '#999999', '#66c2a5', '#fc8d62', '#8da0cb',
                  '#e78ac3', '#a6d854', '#ffd92f', '#e5c494',
                  '#b3b3b3', '#8dd3c7', '#ffffb3', '#bebada',
                  '#fb8072', '#80b1d3', '#fdb462', '#b3de69',
                  '#fccde5', '#d9d9d9', '#bc80bd', '#ccebc5',
                  '#ffed6f', '#1f78b4', '#33a02c', '#fb9a99']
SEED = 42


def load_pretrained_alexnet(device):
    """Load pretrained ImageNet AlexNet."""
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    return model.to(device).eval()


def load_checkpoint_model(checkpoint_path, device):
    """Load model from checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint['model'].to(device).eval()


def extract_layer(model, loader, device, layer=None):
    """Extract features from specified layer (default: fc2)."""
    if layer is None:
        layer = LAYER
    extractor = FeatureExtractor(model, return_nodes={layer: layer})
    extractor.to(device).eval()

    features = []
    with torch.no_grad():
        for images, _ in tqdm(loader, desc=f"Extracting {layer}", leave=False):
            feats = extractor(images.to(device))
            features.append(feats[layer].cpu().numpy())

    return np.vstack(features)


# Alias for backward compatibility
extract_fc2 = extract_layer


def extract_all_layers(model, loader, device, layers=None, conv_pool_size=3):
    """Extract features from all specified layers.

    Args:
        model: PyTorch model
        loader: DataLoader
        device: torch device
        layers: List of layer names (default: ALL_LAYERS)
        conv_pool_size: Spatial size to pool conv features to (default: 3 -> 3x3xC)
                       Set to None to disable pooling (original behavior)

    Returns:
        Dict mapping layer_name -> features array (n_samples x n_features)
    """
    if layers is None:
        layers = ALL_LAYERS

    return_nodes = {layer: layer for layer in layers}
    extractor = FeatureExtractor(model, return_nodes=return_nodes)
    extractor.to(device).eval()

    # Adaptive pooling for conv layers
    if conv_pool_size is not None:
        adaptive_pool = torch.nn.AdaptiveAvgPool2d((conv_pool_size, conv_pool_size))
    else:
        adaptive_pool = None

    # Initialize dict of lists for each layer
    features = {layer: [] for layer in layers}

    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Extracting all layers", leave=False):
            feats = extractor(images.to(device))
            for layer in layers:
                layer_feats = feats[layer]
                # Apply adaptive pooling to conv layers (4D tensors: B x C x H x W)
                if adaptive_pool is not None and layer_feats.dim() == 4:
                    layer_feats = adaptive_pool(layer_feats)
                # Flatten to (B, -1)
                layer_feats = layer_feats.view(layer_feats.size(0), -1)
                features[layer].append(layer_feats.cpu().numpy())

    # Stack all features
    return {layer: np.vstack(features[layer]) for layer in layers}


def load_labels(loader):
    """Load both 32-class PCA labels and semantic labels.

    Returns:
        pca_labels: Array of 32-class PCA labels
        sem_labels: Array of semantic category labels
        synsets: Array of synset IDs (e.g., 'n03729826')
        img_paths: Array of full image paths
    """
    # 32-class PCA labels
    df_pca = pd.read_csv(PCA_LABELS_PATH)
    pca_map = dict(zip(df_pca['image'], df_pca['pca_label']))

    # Semantic labels (10 categories)
    df_sem = pd.read_csv(SEMANTIC_LABELS_PATH)
    sem_map = dict(zip(df_sem['image'], df_sem['pca_label']))

    pca_labels, sem_labels, synsets, img_paths = [], [], [], []
    for sample in loader.dataset.samples:
        # sample structure: (img_path, label, img_id)
        # sample[0] = full image path
        # sample[1] = original label
        # sample[2] = image filename (img_id)
        img_path = sample[0]  # Full path for loading images
        img_name = sample[2]  # Filename for label lookup
        synset = img_name.split('_')[0]  # e.g., n03729826

        pca_labels.append(pca_map.get(img_name, -1))
        sem_labels.append(sem_map.get(img_name, -1))
        synsets.append(synset)

        # Ensure path is absolute
        if not os.path.isabs(img_path):
            img_path = os.path.abspath(img_path)
        img_paths.append(img_path)

    # Debug: Check path validity
    paths_arr = np.array(img_paths)
    n_check = min(10, len(paths_arr))
    n_exist = sum(1 for p in paths_arr[:n_check] if os.path.exists(p))
    print(f"  Image path check: {n_exist}/{n_check} sample paths exist")
    if n_exist == 0 and len(paths_arr) > 0:
        print(f"  Example path: {paths_arr[0]}")

    return (np.array(pca_labels), np.array(sem_labels),
            np.array(synsets), paths_arr)


def load_data_and_models(device=None):
    """Load dataset, labels, and both models. Returns all components needed for analysis.

    Returns:
        feats_list: List of [pretrained_features, 4way_features]
        pca_labels: Array of 32-class labels (filtered)
        sem_labels: Array of semantic labels (filtered)
        synsets: Array of synset IDs (filtered)
        img_paths: Array of image paths (filtered)
        loader: DataLoader
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")

    # Load data
    print(f"\n=== Loading {DATASET} ===")
    cfg = {"dataset": DATASET, "batchsize": 256, "num_workers": 8}
    _, loaders = get_obj_cls_loader(cfg, shuffle=False, train_test_split=False)
    loader = loaders['all']
    print(f"Images: {len(loader.dataset)}")

    # Load labels
    pca_labels, sem_labels, synsets, img_paths = load_labels(loader)
    valid = pca_labels >= 0
    print(f"Valid images: {valid.sum()}/{len(pca_labels)}")

    # Load models
    print("\n=== Loading models ===")
    model_pretrained = load_pretrained_alexnet(device)
    model_32way = load_checkpoint_model(CHECKPOINT_32WAY, device)

    # Extract features
    print("\n=== Extracting FC2 features ===")
    feats_pretrained = extract_fc2(model_pretrained, loader, device)[valid]
    feats_32way = extract_fc2(model_32way, loader, device)[valid]

    # Filter labels
    pca_labels = pca_labels[valid]
    sem_labels = sem_labels[valid]
    synsets = synsets[valid]
    img_paths = img_paths[valid]

    feats_list = [feats_pretrained, feats_32way]

    return feats_list, pca_labels, sem_labels, synsets, img_paths, loader


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data_and_models_all_layers(device=None, layers=None):
    """Load dataset, labels, and extract features from all layers for both models.

    Args:
        device: torch device (default: auto-detect)
        layers: List of layer names (default: ALL_LAYERS)

    Returns:
        feats_dict_pretrained: Dict of layer -> features for pretrained model
        feats_dict_32way: Dict of layer -> features for 32-way model
        pca_labels: Array of 32-class labels (filtered)
        sem_labels: Array of semantic labels (filtered)
        synsets: Array of synset IDs (filtered)
        img_paths: Array of image paths (filtered)
        loader: DataLoader
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if layers is None:
        layers = ALL_LAYERS

    print(f"Device: {device}")

    # Load data
    print(f"\n=== Loading {DATASET} ===")
    cfg = {"dataset": DATASET, "batchsize": 256, "num_workers": 8}
    _, loaders = get_obj_cls_loader(cfg, shuffle=False, train_test_split=False)
    loader = loaders['all']
    print(f"Images: {len(loader.dataset)}")

    # Load labels
    pca_labels, sem_labels, synsets, img_paths = load_labels(loader)
    valid = pca_labels >= 0
    print(f"Valid images: {valid.sum()}/{len(pca_labels)}")

    # Load models
    print("\n=== Loading models ===")
    model_pretrained = load_pretrained_alexnet(device)
    model_32way = load_checkpoint_model(CHECKPOINT_32WAY, device)

    # Extract features from all layers
    print(f"\n=== Extracting features from {len(layers)} layers ===")
    print(f"Layers: {layers}")

    print("\nPretrained model:")
    feats_dict_pretrained = extract_all_layers(model_pretrained, loader, device, layers)

    print("\n32-way model:")
    feats_dict_32way = extract_all_layers(model_32way, loader, device, layers)

    # Filter by valid indices
    feats_dict_pretrained = {layer: feats[valid] for layer, feats in feats_dict_pretrained.items()}
    feats_dict_32way = {layer: feats[valid] for layer, feats in feats_dict_32way.items()}

    # Filter labels
    pca_labels = pca_labels[valid]
    sem_labels = sem_labels[valid]
    synsets = synsets[valid]
    img_paths = img_paths[valid]

    return (feats_dict_pretrained, feats_dict_32way,
            pca_labels, sem_labels, synsets, img_paths, loader)
