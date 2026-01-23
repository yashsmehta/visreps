"""
Variance Ratio Analysis: Within-class vs between-class variance.

Compares cluster tightness between pretrained and 4-way trained models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from utils import (
    load_data_and_models, ensure_output_dir,
    MODEL_NAMES, COLORS_4CLASS, OUTPUT_DIR, SEED
)


def analyze_variance_ratio(feats_list, labels, output_path):
    """Compute within-class vs between-class variance for each model.

    Measures how tight clusters are (within-class distance to centroid)
    compared to how separated they are (between-class centroid distances).

    Returns:
        stats: List of dicts with within, between, and ratio for each model
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    stats = []
    for ax, feats, name in zip(axes, feats_list, MODEL_NAMES):
        # Compute class centroids
        centroids = np.array([feats[labels == c].mean(axis=0) for c in range(4)])
        global_mean = feats.mean(axis=0)

        # Within-class variance (mean distance to class centroid)
        within_var = []
        for c in range(4):
            class_feats = feats[labels == c]
            dists = np.linalg.norm(class_feats - centroids[c], axis=1)
            within_var.append(dists)

        # Between-class variance (centroid distances from global mean)
        between_dists = np.linalg.norm(centroids - global_mean, axis=1)

        # Box plot of within-class distances
        bp = ax.boxplot(within_var, labels=[f'Class {i}' for i in range(4)], patch_artist=True)
        for patch, color in zip(bp['boxes'], COLORS_4CLASS):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Add between-class info
        mean_within = np.mean([np.mean(w) for w in within_var])
        mean_between = np.mean(between_dists)
        ratio = mean_between / mean_within if mean_within > 0 else 0

        ax.set_xlabel('Class', fontsize=11)
        ax.set_ylabel('Distance to Class Centroid', fontsize=11)
        ax.set_title(f'{name}\nB/W Ratio: {ratio:.2f}', fontsize=13, fontweight='bold')
        ax.set_facecolor('#FAFAFA')

        stats.append({
            'name': name,
            'within': mean_within,
            'between': mean_between,
            'ratio': ratio
        })
        print(f"  {name}: Within={mean_within:.2f}, Between={mean_between:.2f}, Ratio={ratio:.2f}")

    plt.suptitle('Cluster Tightness: Distance to Class Centroid',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()

    return stats


def main():
    np.random.seed(SEED)
    ensure_output_dir()

    print("=" * 60)
    print("Variance Ratio Analysis")
    print("=" * 60)

    feats_list, pca_labels, _, _, _, _ = load_data_and_models()

    output_path = os.path.join(OUTPUT_DIR, "variance_ratio.png")
    stats = analyze_variance_ratio(feats_list, pca_labels, output_path)

    print("\nDone!")
    return stats


if __name__ == "__main__":
    main()
