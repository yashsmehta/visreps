"""
PCA Visualization: Project features onto PC1-PC2, color by hierarchical PCA labels.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Configuration
MODEL_NAME = "alexnet"
FEATURES_PATH = f"datasets/obj_cls/imagenet/features_{MODEL_NAME}.npz"
EIGENVECTORS_PATH = f"datasets/obj_cls/imagenet/eigenvectors_{MODEL_NAME}.npz"
LABELS_DIR = f"pca_labels/pca_labels_{MODEL_NAME}_hierarchical"
N_CLASSES = 4  # 2, 4, 8, 16, 32, or 64
SAMPLE_FRACTION = 0.05
SEED = 42
OUTPUT_DIR = "experiments/results"


def load_data():
    """Load eigenvectors, features, and labels."""
    print(f"Loading eigenvectors from {EIGENVECTORS_PATH}")
    pca = np.load(EIGENVECTORS_PATH)
    eigenvectors, mean = pca['eigenvectors'][:, :4], pca['mean']

    print(f"Loading features from {FEATURES_PATH}")
    data = np.load(FEATURES_PATH, allow_pickle=True)
    names = data['image_names']
    if names.size > 0 and isinstance(names[0], (bytes, np.bytes_)):
        names = np.array([n.decode('utf-8') for n in names])
    names = np.array([os.path.basename(str(n)) for n in names])

    for key in ['fc2', 'clip_features', 'features', 'dreamsim_features']:
        if key in data:
            features = data[key].reshape(len(names), -1)
            break

    print(f"Loading labels from {LABELS_DIR}/n_classes_{N_CLASSES}.csv")
    labels_df = pd.read_csv(f"{LABELS_DIR}/n_classes_{N_CLASSES}.csv")
    name_to_label = dict(zip(labels_df['image'], labels_df['pca_label']))
    labels = np.array([name_to_label[n] for n in names])

    # Sample
    np.random.seed(SEED)
    n_samples = int(len(names) * SAMPLE_FRACTION)
    sample_idx = np.random.choice(len(names), n_samples, replace=False)
    print(f"Sampled {n_samples:,} points ({SAMPLE_FRACTION*100:.0f}%)")

    scores = (features[sample_idx] - mean) @ eigenvectors
    sample_labels = labels[sample_idx]
    return scores, sample_labels


def plot_scatter():
    """PC1-PC2 scatter plot."""
    scores, sample_labels = load_data()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Spectral(np.linspace(0.05, 0.95, N_CLASSES))

    for c in range(N_CLASSES):
        mask = sample_labels == c
        ax.scatter(scores[mask, 0], scores[mask, 1], c=[colors[c]], 
                   label=f'Class {c} (n={mask.sum():,})', alpha=0.6, s=10, edgecolors='none')

    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title(f'AlexNet fc2 Features on PC1-PC2 ({N_CLASSES} hierarchical classes)', fontsize=14)
    ax.legend(loc='best', fontsize=9, ncol=2 if N_CLASSES > 4 else 1, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_facecolor('#FAFAFA')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = f"{OUTPUT_DIR}/pca_pc1pc2_{N_CLASSES}classes.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_path}")
    plt.close()


def plot_1d_distributions():
    """2x2 grid showing 1D distributions along PC1, PC2, PC3, PC4 (all samples)."""
    scores, _ = load_data()
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    colors = ['#E24A33', '#348ABD', '#988ED5', '#8EBA42']  # distinct colors per PC
    
    for i, ax in enumerate(axes.flat):
        ax.hist(scores[:, i], bins=80, alpha=0.7, color=colors[i], density=True)
        ax.set_xlabel(f'PC{i+1}', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'Distribution along PC{i+1}', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_facecolor('#FAFAFA')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = f"{OUTPUT_DIR}/pca_1d_distributions.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved to {output_path}")
    plt.close()


def main():
    plot_scatter()
    plot_1d_distributions()


if __name__ == "__main__":
    main()
