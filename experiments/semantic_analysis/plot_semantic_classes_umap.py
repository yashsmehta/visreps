"""
2D Visualization: Compare FC2 representations across models trained with 2-1000 classes.
Uses UMAP for dimensionality reduction, colored by semantic super-categories.
"""

import os
import sys
import warnings

# Get project root (visreps/) and add to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Ensure .env is loaded from project root
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

# Suppress UMAP n_jobs warning when using random_state
warnings.filterwarnings("ignore", message="n_jobs value.*overridden to 1 by setting random_state")

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import umap
from tqdm import tqdm

from visreps.models.utils import FeatureExtractor
from visreps.dataloaders.obj_cls import get_obj_cls_loader

# Import category definitions from make_semantic_labels.py
from experiments.wordnet.make_semantic_labels import SUPER_CATEGORIES

# =============================================================================
# Configuration
# =============================================================================
# Models to compare (2x4 grid = 8 models)
model = "clip"
MODELS = [
    {"name": "2-way", "dir": f"/data/ymehta3/{model}_pca/cfg2a"},
    {"name": "4-way", "dir": f"/data/ymehta3/{model}_pca/cfg4a"},
    {"name": "8-way", "dir": f"/data/ymehta3/{model}_pca/cfg8a"},
    {"name": "16-way", "dir": f"/data/ymehta3/{model}_pca/cfg16a"},
    {"name": "32-way", "dir": f"/data/ymehta3/{model}_pca/cfg32a"},
    {"name": "64-way", "dir": f"/data/ymehta3/{model}_pca/cfg64a"},
    {"name": "128-way", "dir": f"/data/ymehta3/{model}_pca/cfg128a"},
    {"name": "1000-way", "dir": f"/data/ymehta3/imagenet1k/cfg1000a"},
]

CHECKPOINT_FILE = "checkpoint_epoch_20.pth"

DATASET = "imagenet-mini-50"
WORDNET_LABELS_PATH = os.path.join(PROJECT_ROOT, "experiments/wordnet/semantic_categories.csv")
LAYER = "fc2"

# Category names derived from SUPER_CATEGORIES (same order as labels)
CATEGORY_NAMES = list(SUPER_CATEGORIES.keys())

# Generate distinct colors for categories using a colormap
def generate_category_colors(n_categories):
    """Generate n distinct colors using matplotlib colormaps."""
    if n_categories <= 10:
        # Use tab10 for up to 10 categories
        cmap = plt.cm.tab10
    elif n_categories <= 20:
        # Use tab20 for up to 20 categories
        cmap = plt.cm.tab20
    else:
        # Use a continuous colormap for more categories
        cmap = plt.cm.nipy_spectral
    return [cmap(i / max(n_categories - 1, 1)) for i in range(n_categories)]

CATEGORY_COLORS = generate_category_colors(len(CATEGORY_NAMES))

KNN_NEIGHBORS = 50
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "experiments/semantic_analysis")
SEED = 42

# Preprocessing options
L2_NORMALIZE = True

# Visualization options
ZOOM_PERCENTILE = 2
POINT_SIZE = 2
POINT_ALPHA = 0.5


def load_model(checkpoint_dir, device):
    """Load model from checkpoint."""
    checkpoint_path = os.path.join(checkpoint_dir, CHECKPOINT_FILE)
    if not os.path.exists(checkpoint_path):
        print(f"  Warning: Checkpoint not found: {checkpoint_path}")
        return None
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = checkpoint['model'].to(device)
    model.eval()
    return model


def extract_features(model, loader, device):
    """Extract features using FeatureExtractor."""
    extractor = FeatureExtractor(model, return_nodes={LAYER: LAYER})
    extractor.to(device).eval()
    
    features = []
    with torch.no_grad():
        for images, _ in tqdm(loader, desc=f"Extracting {LAYER}", leave=False):
            images = images.to(device)
            feats = extractor(images)
            features.append(feats[LAYER].cpu().numpy())
    
    return np.vstack(features)


def load_wordnet_labels(loader):
    """Load WordNet labels and match to dataset images."""
    df = pd.read_csv(WORDNET_LABELS_PATH)
    label_map = dict(zip(df['image'], df['pca_label']))
    
    labels = []
    for sample in loader.dataset.samples:
        img_id = os.path.basename(sample[2])
        labels.append(label_map.get(img_id, -1))
    
    return np.array(labels)


def preprocess_features(features):
    """L2-normalize features."""
    if L2_NORMALIZE:
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        features = features / norms
    return features


def apply_umap(features, n_neighbors=KNN_NEIGHBORS):
    """Apply UMAP dimensionality reduction."""
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric='cosine',
        random_state=SEED,
        verbose=False
    )
    return reducer.fit_transform(features.astype(np.float32))


def plot_grid(all_coords, labels, model_names, output_path):
    """Plot 2x4 grid of UMAP visualizations with shared legend."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    unique_labels = np.unique(labels[labels >= 0])
    
    for idx, (coords, name) in enumerate(zip(all_coords, model_names)):
        ax = axes[idx]
        
        if coords is None:
            ax.text(0.5, 0.5, f'{name}\n(not available)', 
                    ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            continue
        
        # Plot each category
        for label in unique_labels:
            mask = labels == label
            color = CATEGORY_COLORS[label] if label < len(CATEGORY_COLORS) else '#000000'
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=color, 
                alpha=POINT_ALPHA, 
                s=POINT_SIZE, 
                edgecolors='none',
                rasterized=True  # For faster rendering
            )
        
        # Zoom to central distribution
        if ZOOM_PERCENTILE is not None:
            xlim = np.percentile(coords[:, 0], [ZOOM_PERCENTILE, 100 - ZOOM_PERCENTILE])
            ylim = np.percentile(coords[:, 1], [ZOOM_PERCENTILE, 100 - ZOOM_PERCENTILE])
            xpad = (xlim[1] - xlim[0]) * 0.1
            ypad = (ylim[1] - ylim[0]) * 0.1
            ax.set_xlim(xlim[0] - xpad, xlim[1] + xpad)
            ax.set_ylim(ylim[0] - ypad, ylim[1] + ypad)
        
        ax.set_xlabel('UMAP 1', fontsize=10)
        ax.set_ylabel('UMAP 2', fontsize=10)
        ax.set_title(f'{name}', fontsize=14, fontweight='bold')
        ax.set_facecolor('#FAFAFA')
        ax.tick_params(labelsize=8)
    
    # Create legend with visible colored dots
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=CATEGORY_COLORS[i], 
               markersize=10, label=CATEGORY_NAMES[i])
        for i in range(len(CATEGORY_NAMES))
    ]
    
    fig.legend(
        handles=legend_elements,
        loc='center right',
        bbox_to_anchor=(0.99, 0.5),
        fontsize=11,
        title='Semantic Category',
        title_fontsize=12,
        frameon=True,
        fancybox=True,
        shadow=True
    )
    
    plt.suptitle('UMAP Visualization of FC2 Features Across Training Granularities', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved to {output_path}")
    plt.close()


def main():
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    print(f"\n=== Loading {DATASET} ===")
    cfg = {"dataset": DATASET, "batchsize": 256, "num_workers": 8}
    _, loaders = get_obj_cls_loader(cfg, shuffle=False, train_test_split=False)
    loader = loaders['all']
    print(f"Loaded {len(loader.dataset)} images")
    
    # Load WordNet labels
    print("\n=== Loading WordNet labels ===")
    labels = load_wordnet_labels(loader)
    valid_mask = labels >= 0
    print(f"Matched {valid_mask.sum()} / {len(labels)} images to WordNet labels")
    
    # Extract features from all models
    all_features = []
    model_names = []
    
    for model_info in MODELS:
        name = model_info["name"]
        model_dir = model_info["dir"]
        model_names.append(name)
        
        print(f"\n=== Processing {name} model ===")
        model = load_model(model_dir, device)
        
        if model is None:
            all_features.append(None)
            continue
        
        features = extract_features(model, loader, device)
        del model
        torch.cuda.empty_cache()
        
        # Filter to valid samples and preprocess
        features = features[valid_mask]
        features = preprocess_features(features)
        all_features.append(features)
        print(f"  Features shape: {features.shape}")
    
    # Apply UMAP to each
    print("\n=== Applying UMAP to all models ===")
    all_coords = []
    for name, features in zip(model_names, all_features):
        if features is None:
            all_coords.append(None)
            continue
        print(f"  {name}...")
        coords = apply_umap(features)
        all_coords.append(coords)
    
    # Plot
    print("\n=== Creating visualization ===")
    output_path = os.path.join(OUTPUT_DIR, f"semantic_classes_umap_{model}.png")
    plot_grid(all_coords, labels[valid_mask], model_names, output_path)


if __name__ == "__main__":
    main()
