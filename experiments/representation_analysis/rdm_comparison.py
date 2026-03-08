"""
RSM Comparison: Representational Similarity Matrix analysis.

Computes and visualizes RSMs, comparing correlation between models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr

from utils import (
    load_data_and_models, ensure_output_dir,
    MODEL_NAMES, OUTPUT_DIR, SEED
)


def analyze_rsm(feats_list, labels, output_path, n_samples=500):
    """Compute and visualize RSMs, compare correlation between models.

    Args:
        feats_list: List of feature arrays for each model
        labels: 4-class PCA labels for stratified sampling
        output_path: Where to save the figure
        n_samples: Number of images to sample
    """
    np.random.seed(SEED)

    # Sample images (stratified by class)
    idx_sample = []
    for c in range(4):
        class_idx = np.where(labels == c)[0]
        n_per_class = min(n_samples // 4, len(class_idx))
        idx_sample.extend(np.random.choice(class_idx, n_per_class, replace=False))
    idx_sample = np.array(idx_sample)

    # Sort by class for visualization
    sort_order = np.argsort(labels[idx_sample])
    idx_sample = idx_sample[sort_order]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    rsms = []
    for ax, feats, name in zip(axes[:2], feats_list, MODEL_NAMES):
        feats_sub = feats[idx_sample]
        rsm = cosine_similarity(feats_sub)
        rsms.append(rsm)

        im = ax.imshow(rsm, cmap='RdBu_r', vmin=-0.5, vmax=1)
        ax.set_title(f'{name}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Image', fontsize=10)
        ax.set_ylabel('Image', fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)

    # RSM correlation
    triu_idx = np.triu_indices(len(idx_sample), k=1)
    rsm1_flat = rsms[0][triu_idx]
    rsm2_flat = rsms[1][triu_idx]
    corr, _ = spearmanr(rsm1_flat, rsm2_flat)

    axes[2].scatter(rsm1_flat[::10], rsm2_flat[::10], alpha=0.3, s=1)
    axes[2].plot([-0.5, 1], [-0.5, 1], 'r--', linewidth=1)
    axes[2].set_xlabel(f'{MODEL_NAMES[0]} similarity', fontsize=11)
    axes[2].set_ylabel(f'{MODEL_NAMES[1]} similarity', fontsize=11)
    axes[2].set_title(f'RSM Correlation: r={corr:.3f}', fontsize=13, fontweight='bold')
    axes[2].set_facecolor('#FAFAFA')

    plt.suptitle('Representational Similarity Matrices (500 images, sorted by class)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    print(f"  RSM Spearman correlation: {corr:.4f}")
    plt.close()

    return corr


def main():
    np.random.seed(SEED)
    ensure_output_dir()

    print("=" * 60)
    print("RSM Comparison Analysis")
    print("=" * 60)

    feats_list, pca_labels, _, _, _, _ = load_data_and_models()

    output_path = os.path.join(OUTPUT_DIR, "rsm_comparison.png")
    corr = analyze_rsm(feats_list, pca_labels, output_path)

    print("\nDone!")
    return corr


if __name__ == "__main__":
    main()
