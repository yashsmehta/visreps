"""
Visualize the distribution of images in PCA space.

Creates:
1. Four 1D density plots for PC1, PC2, PC3, PC4
2. One 2D hexbin plot for PC1 vs PC2
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# Configuration (matching make_pca_classes.py)
FEATURES_PATH = "datasets/obj_cls/imagenet/features_alexnet.npz"
N_COMPONENTS = 4  # We need 4 PCs for visualization

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['figure.titlesize'] = 14

def load_and_transform_features():
    """Load features and transform to PCA space."""
    print(f"Loading features from: {FEATURES_PATH}")
    data_dict = np.load(FEATURES_PATH, allow_pickle=True)

    # Detect feature type
    feature_keys = {'fc2': True, 'clip_features': True, 'features': True}
    for key in feature_keys:
        if key in data_dict:
            feature_array = data_dict[key]
            print(f"Found features under key: '{key}'")
            break
    else:
        raise ValueError(f"No valid feature key found. Available: {list(data_dict.keys())}")

    n_samples = feature_array.shape[0]
    print(f"Total samples: {n_samples:,}")

    # Reshape to 2D if needed
    features_2d = feature_array.reshape(n_samples, -1) if feature_array.ndim != 2 else feature_array
    print(f"Feature shape: {features_2d.shape}")

    # Standardize features
    print("Standardizing features...")
    features_scaled = StandardScaler().fit_transform(features_2d)

    # Fit PCA on subset (matching make_pca_classes.py)
    n_fit = min(110000, n_samples)
    np.random.seed(42)
    fit_indices = np.random.choice(n_samples, n_fit, replace=False)

    print(f"Fitting PCA with {N_COMPONENTS} components on {n_fit:,} samples...")
    pca = PCA(n_components=N_COMPONENTS)
    pca.fit(features_scaled[fit_indices])

    # Print variance explained
    print("\nVariance explained by each PC:")
    for i, var in enumerate(pca.explained_variance_ratio_, 1):
        print(f"  PC{i}: {var*100:.2f}%")
    print(f"  Total: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")

    # Transform all samples
    print(f"\nTransforming all {n_samples:,} samples to PCA space...")
    pc_scores = pca.transform(features_scaled)

    return pc_scores, pca


def plot_1d_densities(pc_scores):
    """Create 4 horizontal subplots showing density for each PC."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle('Distribution of Images along Principal Components', fontsize=16, y=1.02)

    colors = ['#E63946', '#457B9D', '#2A9D8F', '#E9C46A']

    for i, (ax, color) in enumerate(zip(axes, colors)):
        pc_data = pc_scores[:, i]

        # Plot histogram with narrower bins
        ax.hist(pc_data, bins=200, density=True, alpha=0.7, color=color, edgecolor='black', linewidth=0.3)

        # Add median line
        median = np.median(pc_data)
        ax.axvline(median, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Median: {median:.2f}')

        # Formatting
        ax.set_xlabel(f'PC{i+1} Score', fontweight='bold')
        ax.set_ylabel('Density' if i == 0 else '')
        ax.set_title(f'PC{i+1}', fontweight='bold', fontsize=14)
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        # Add statistics text
        stats_text = f'μ={pc_data.mean():.2f}\nσ={pc_data.std():.2f}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                fontsize=9)

    plt.tight_layout()
    output_path = 'pca_1d_distributions.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved 1D density plots to: {output_path}")
    plt.close()


def plot_2d_hexbin(pc_scores):
    """Create 2D hexbin plot for PC1 vs PC2."""
    fig, ax = plt.subplots(figsize=(10, 9))

    pc1 = pc_scores[:, 0]
    pc2 = pc_scores[:, 1]

    # Create hexbin plot
    hexbin = ax.hexbin(pc1, pc2, gridsize=80, cmap='YlOrRd', mincnt=1, linewidths=0.2)

    # Add colorbar
    cbar = plt.colorbar(hexbin, ax=ax)
    cbar.set_label('Number of Images', rotation=270, labelpad=20, fontsize=12, fontweight='bold')

    # Add median lines
    median_pc1 = np.median(pc1)
    median_pc2 = np.median(pc2)
    ax.axvline(median_pc1, color='blue', linestyle='--', linewidth=2, alpha=0.6, label=f'PC1 median: {median_pc1:.2f}')
    ax.axhline(median_pc2, color='green', linestyle='--', linewidth=2, alpha=0.6, label=f'PC2 median: {median_pc2:.2f}')

    # Formatting
    ax.set_xlabel('PC1 Score', fontsize=13, fontweight='bold')
    ax.set_ylabel('PC2 Score', fontsize=13, fontweight='bold')
    ax.set_title('2D Distribution: PC1 vs PC2', fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add statistics text
    stats_text = (f'PC1: μ={pc1.mean():.2f}, σ={pc1.std():.2f}\n'
                  f'PC2: μ={pc2.mean():.2f}, σ={pc2.std():.2f}\n'
                  f'Total images: {len(pc1):,}')
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
            fontsize=10)

    plt.tight_layout()
    output_path = 'pca_2d_hexbin.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved 2D hexbin plot to: {output_path}")
    plt.close()


def main():
    print("="*60)
    print("PCA Distribution Visualization")
    print("="*60)

    # Load and transform
    pc_scores, pca = load_and_transform_features()

    print("\n" + "="*60)
    print("Creating visualizations...")
    print("="*60)

    # Create plots
    plot_1d_densities(pc_scores)
    plot_2d_hexbin(pc_scores)

    print("\n" + "="*60)
    print("Done! All plots saved successfully.")
    print("="*60)


if __name__ == '__main__':
    main()
