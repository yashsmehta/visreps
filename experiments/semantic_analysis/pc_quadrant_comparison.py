"""
Compare PC1-PC2 space between two models.

Creates side-by-side scatter plots:
- Left: Model1 features projected onto Model1 PCs, colored by median-split quadrants
- Right: Model2 features projected onto Model2 PCs, colored by Model1 quadrant assignments

This reveals whether the two models organize images similarly along their top PCs.

Usage:
    python experiments/semantic_analysis/pc_quadrant_comparison.py --model1 alexnet --model2 vit
    python experiments/semantic_analysis/pc_quadrant_comparison.py --model1 dino --model2 clip
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))


def load_features(model_name, dataset='imagenet-mini-50'):
    """Load features and image names for a model."""
    for pattern in [f"features_{model_name}.npz", f"features_{model_name}_features.npz"]:
        features_path = f"datasets/obj_cls/{dataset}/{pattern}"
        if os.path.exists(features_path):
            break

    features_data = np.load(features_path, allow_pickle=True)
    features_key = [k for k in features_data.keys() if 'features' in k and k != 'image_names'][0]

    names = features_data['image_names']
    if names.size > 0 and isinstance(names[0], (bytes, np.bytes_)):
        names = np.array([n.decode('utf-8') for n in names])

    features = features_data[features_key].reshape(len(names), -1)
    return features, names


def load_eigenvectors(model_name):
    """Load eigenvectors and mean from full ImageNet PCA."""
    eigenvectors_path = f"datasets/obj_cls/imagenet/eigenvectors_{model_name}.npz"
    pca_data = np.load(eigenvectors_path)
    return pca_data['eigenvectors'], pca_data['mean']


def project_onto_pcs(features, eigenvectors, mean, n_pcs=2):
    """Project features onto first n principal components."""
    centered = features - mean
    return centered @ eigenvectors[:, :n_pcs]


def assign_quadrants(pc1_scores, pc2_scores):
    """Assign quadrant based on median splits of PC1 and PC2.

    Color scheme logic:
        - PC1 determines color family: low=cool (blues), high=warm (corals)
        - PC2 determines shade: low=light, high=dark

    Returns array of quadrant indices (0-3):
        Q0: low PC1, low PC2  -> light blue
        Q1: low PC1, high PC2 -> dark blue
        Q2: high PC1, low PC2 -> light coral
        Q3: high PC1, high PC2 -> dark coral
    """
    pc1_median = np.median(pc1_scores)
    pc2_median = np.median(pc2_scores)

    quadrants = np.zeros(len(pc1_scores), dtype=int)

    low_pc1 = pc1_scores <= pc1_median
    high_pc1 = pc1_scores > pc1_median
    low_pc2 = pc2_scores <= pc2_median
    high_pc2 = pc2_scores > pc2_median

    quadrants[low_pc1 & low_pc2] = 0
    quadrants[low_pc1 & high_pc2] = 1
    quadrants[high_pc1 & low_pc2] = 2
    quadrants[high_pc1 & high_pc2] = 3

    return quadrants, pc1_median, pc2_median


def main():
    parser = argparse.ArgumentParser(description="Compare PC space between two models")
    parser.add_argument('--model1', type=str, default='alexnet',
                        choices=['alexnet', 'vit', 'clip', 'dino'],
                        help='First model (defines quadrant coloring)')
    parser.add_argument('--model2', type=str, default='vit',
                        choices=['alexnet', 'vit', 'clip', 'dino'],
                        help='Second model (colored by model1 quadrants)')
    parser.add_argument('--dataset', type=str, default='imagenet-mini-50',
                        choices=['imagenet', 'imagenet-mini-50'])
    args = parser.parse_args()

    model1, model2 = args.model1, args.model2

    # Load Model1 data
    print(f"Loading {model1} features...")
    model1_features, model1_names = load_features(model1, args.dataset)
    model1_eigenvectors, model1_mean = load_eigenvectors(model1)

    # Load Model2 data
    print(f"Loading {model2} features...")
    model2_features, model2_names = load_features(model2, args.dataset)
    model2_eigenvectors, model2_mean = load_eigenvectors(model2)

    print(f"{model1}: {len(model1_names)} images")
    print(f"{model2}: {len(model2_names)} images")

    # Project onto PCs
    model1_pcs = project_onto_pcs(model1_features, model1_eigenvectors, model1_mean)
    model2_pcs = project_onto_pcs(model2_features, model2_eigenvectors, model2_mean)

    # Assign quadrants based on Model1 median splits
    model1_quadrants, pc1_med, pc2_med = assign_quadrants(model1_pcs[:, 0], model1_pcs[:, 1])

    # Create mapping from image name (basename) to Model1 quadrant
    name_to_quadrant = {}
    for idx, name in enumerate(model1_names):
        basename = os.path.basename(name)
        name_to_quadrant[basename] = model1_quadrants[idx]

    # Get Model2 colors based on Model1 quadrant assignments
    model2_colors = np.zeros(len(model2_names), dtype=int)
    matched = 0
    for idx, name in enumerate(model2_names):
        basename = os.path.basename(name)
        if basename in name_to_quadrant:
            model2_colors[idx] = name_to_quadrant[basename]
            matched += 1
        else:
            model2_colors[idx] = -1

    print(f"Matched {matched}/{len(model2_names)} images between {model1} and {model2}")

    valid_mask = model2_colors >= 0
    if not valid_mask.all():
        print(f"Warning: {(~valid_mask).sum()} {model2} images not found in {model1}")

    # Color scheme: PC1 splits into families, PC2 gives contrast within family
    # Cool family (low PC1): teal & indigo | Warm family (high PC1): gold & crimson
    colors = [
        '#1b9e77',  # Q0: low PC1, low PC2  -> teal
        '#7570b3',  # Q1: low PC1, high PC2 -> indigo/purple
        '#e6ab02',  # Q2: high PC1, low PC2 -> gold/amber
        '#d62728',  # Q3: high PC1, high PC2 -> crimson
    ]

    # Set up Nature-style plotting
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 10,
        'axes.linewidth': 0.8,
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
    })

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Plot Model1 (left)
    for q in range(4):
        mask = model1_quadrants == q
        ax1.scatter(model1_pcs[mask, 0], model1_pcs[mask, 1],
                    c=colors[q], alpha=0.35, s=3, edgecolors='none', rasterized=True)
    ax1.axhline(y=0, color='#666666', linestyle='-', alpha=0.3, linewidth=0.5, zorder=0)
    ax1.axvline(x=0, color='#666666', linestyle='-', alpha=0.3, linewidth=0.5, zorder=0)
    ax1.axhline(y=pc2_med, color='#333333', linestyle='--', alpha=0.6, linewidth=0.8)
    ax1.axvline(x=pc1_med, color='#333333', linestyle='--', alpha=0.6, linewidth=0.8)
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    ax1.set_title(model1.upper(), fontweight='medium')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Plot Model2 with Model1 colors (right)
    for q in range(4):
        mask = (model2_colors == q) & valid_mask
        ax2.scatter(model2_pcs[mask, 0], model2_pcs[mask, 1],
                    c=colors[q], alpha=0.35, s=3, edgecolors='none', rasterized=True)
    ax2.axhline(y=0, color='#666666', linestyle='-', alpha=0.3, linewidth=0.5, zorder=0)
    ax2.axvline(x=0, color='#666666', linestyle='-', alpha=0.3, linewidth=0.5, zorder=0)
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_title(f'{model2.upper()} ({model1} coloring)', fontweight='medium')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout(w_pad=2)

    output_dir = 'experiments/semantic_analysis'
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'pc_quadrant_{model1}_vs_{model2}.png')
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved figure to {output_path}")


if __name__ == '__main__':
    main()
