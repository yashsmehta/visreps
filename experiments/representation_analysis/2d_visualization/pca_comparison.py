"""
PCA Comparison: Pretrained vs 4-way trained AlexNet FC2 representations.

Subplot 1: TorchVision pretrained AlexNet (source of PCA labels)
Subplot 2: AlexNet trained on 4-class PCA labels

Both colored by the same 4-class labels from pca_labels_alexnet/n_classes_4.csv
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm

from visreps.models.utils import FeatureExtractor
from visreps.dataloaders.obj_cls import get_obj_cls_loader
import torchvision.models as models

# =============================================================================
# Configuration
# =============================================================================
DATASET = "imagenet-mini-50"
LABELS_PATH = "pca_labels/pca_labels_alexnet/n_classes_4.csv"
LAYER = "fc2"
CHECKPOINT_4WAY = "/data/ymehta3/alexnet_pca/cfg4a/checkpoint_epoch_20.pth"

# Colors for 4 classes
COLORS = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
CLASS_NAMES = ['Class 0', 'Class 1', 'Class 2', 'Class 3']

OUTPUT_PATH = "experiments/2d_visualization/pca_comparison_pretrained_vs_4way.png"


def load_pretrained_alexnet(device):
    """Load TorchVision pretrained AlexNet."""
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    return model.to(device).eval()


def load_checkpoint_model(checkpoint_path, device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = checkpoint['model']
    return model.to(device).eval()


def extract_fc2(model, loader, device):
    """Extract FC2 features."""
    extractor = FeatureExtractor(model, return_nodes={LAYER: LAYER})
    extractor.to(device).eval()
    
    features = []
    with torch.no_grad():
        for images, _ in tqdm(loader, desc=f"Extracting {LAYER}", leave=False):
            feats = extractor(images.to(device))
            features.append(feats[LAYER].cpu().numpy())
    
    return np.vstack(features)


def load_labels(loader):
    """Load 4-class PCA labels for dataset images."""
    df = pd.read_csv(LABELS_PATH)
    label_map = dict(zip(df['image'], df['pca_label']))
    
    labels = []
    for sample in loader.dataset.samples:
        img_name = os.path.basename(sample[2])
        labels.append(label_map.get(img_name, -1))
    
    return np.array(labels)


def plot_comparison(coords1, coords2, labels, output_path):
    """Plot side-by-side PCA projections."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    titles = ['Pretrained AlexNet (TorchVision)', '4-way Trained AlexNet']
    
    for ax, coords, title in zip(axes, [coords1, coords2], titles):
        for class_idx in range(4):
            mask = labels == class_idx
            ax.scatter(
                coords[mask, 0], coords[mask, 1],
                c=COLORS[class_idx], label=CLASS_NAMES[class_idx],
                alpha=0.6, s=10, edgecolors='none'
            )
        
        ax.set_xlabel('PC1', fontsize=12)
        ax.set_ylabel('PC2', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.set_facecolor('#FAFAFA')
    
    plt.suptitle('FC2 Representations: PC1 vs PC2 (colored by 4-class PCA labels)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load data
    print(f"\n=== Loading {DATASET} ===")
    cfg = {"dataset": DATASET, "batchsize": 256, "num_workers": 8}
    _, loaders = get_obj_cls_loader(cfg, shuffle=False, train_test_split=False)
    loader = loaders['all']
    print(f"Images: {len(loader.dataset)}")
    
    # Load labels
    labels = load_labels(loader)
    valid = labels >= 0
    print(f"Matched labels: {valid.sum()}/{len(labels)}")
    
    # Load models
    print("\n=== Loading models ===")
    model_pretrained = load_pretrained_alexnet(device)
    model_4way = load_checkpoint_model(CHECKPOINT_4WAY, device)
    
    # Extract features
    print("\n=== Extracting FC2 features ===")
    feats_pretrained = extract_fc2(model_pretrained, loader, device)[valid]
    feats_4way = extract_fc2(model_4way, loader, device)[valid]
    
    # PCA projection
    print("\n=== Computing PCA ===")
    pca1 = PCA(n_components=2).fit_transform(feats_pretrained)
    pca2 = PCA(n_components=2).fit_transform(feats_4way)
    
    # Plot
    print("\n=== Creating visualization ===")
    plot_comparison(pca1, pca2, labels[valid], OUTPUT_PATH)


if __name__ == "__main__":
    main()
