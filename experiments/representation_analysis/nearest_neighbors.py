"""
Nearest Neighbor Retrieval Analysis.

Shows k-nearest neighbors for query images, comparing pretrained vs 4-way models.
Green border = same class, Red border = different class.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

from utils import (
    load_data_and_models, ensure_output_dir,
    MODEL_NAMES, OUTPUT_DIR, SEED
)


def load_image_safe(img_path, size=(224, 224)):
    """Safely load and resize an image, with detailed error reporting.

    Args:
        img_path: Full path to image file
        size: Tuple of (width, height) for resizing

    Returns:
        PIL Image or None if loading fails
        Error message string if failed, empty string if success
    """
    if img_path is None:
        return None, "Path is None"

    if not isinstance(img_path, str):
        return None, f"Not string: {type(img_path).__name__}"

    # Convert to string and ensure absolute path
    img_path = str(img_path)
    if not os.path.isabs(img_path):
        img_path = os.path.abspath(img_path)

    if not os.path.exists(img_path):
        # Try common path variations
        basename = os.path.basename(img_path)
        return None, f"Not found: {basename[:20]}"

    try:
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = img.resize(size)
        return img, ""
    except Exception as e:
        return None, str(e)[:30]


def analyze_nearest_neighbors(feats_list, labels, synsets, img_paths, output_path,
                               n_queries=4, k=5):
    """Show k-nearest neighbors for query images.

    Args:
        feats_list: List of feature arrays for each model
        labels: 4-class PCA labels
        synsets: Array of synset IDs (e.g., 'n03729826')
        img_paths: Array of full image file paths
        output_path: Where to save the figure
        n_queries: Number of query images (one per class)
        k: Number of nearest neighbors to show

    Returns:
        retrieval_stats: Dict with retrieval accuracy for each model
    """
    np.random.seed(SEED)

    # Convert img_paths to list of strings if needed
    img_paths = np.array([str(p) for p in img_paths])

    # Debug: Check image paths
    print(f"\n  Verifying image paths...")
    print(f"  Total images: {len(img_paths)}")

    # Sample check - try more samples to find valid paths
    n_check = min(20, len(img_paths))
    paths_exist = 0
    valid_indices = []
    for i in range(n_check):
        path = str(img_paths[i])
        exists = os.path.exists(path)
        if exists:
            paths_exist += 1
            valid_indices.append(i)
        if i < 3:
            print(f"    Path {i}: {path[:80]}...")
            print(f"           exists={exists}")

    print(f"  Valid paths: {paths_exist}/{n_check} checked")

    if paths_exist == 0:
        print(f"  WARNING: None of the sampled paths exist!")
        print(f"  First path: {img_paths[0]}")
        print(f"  Creating visualization with placeholder images...")

    # Pick query images (one per class, prefer valid images if available)
    query_idx = []
    for c in range(min(4, n_queries)):
        class_idx = np.where(labels == c)[0]
        if len(class_idx) > 0:
            # Try to find a valid image path
            found_valid = False
            for candidate in np.random.permutation(class_idx)[:20]:
                if os.path.exists(str(img_paths[candidate])):
                    query_idx.append(candidate)
                    found_valid = True
                    break
            if not found_valid:
                # Fallback: just pick one even if path doesn't exist
                query_idx.append(np.random.choice(class_idx))

    if len(query_idx) == 0:
        print("  ERROR: No query images found (no samples in any class)!")
        return None

    print(f"  Using {len(query_idx)} query images (one per class)")

    # Create figure with correct dimensions
    n_rows = len(query_idx)
    n_cols = 2 * (k + 1)  # 2 models, each with query + k neighbors

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.2 * n_cols, 2.5 * n_rows))

    # Handle case where we have only one row
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Track image loading stats
    images_loaded = 0
    images_failed = 0

    retrieval_stats = {'pretrained': [], '4way': []}

    for row, q_idx in enumerate(query_idx):
        for model_idx, (feats, name) in enumerate(zip(feats_list, MODEL_NAMES)):
            # Compute similarities
            query_feat = feats[q_idx:q_idx+1]
            sims = cosine_similarity(query_feat, feats)[0]
            sims[q_idx] = -np.inf  # Exclude self
            top_k = np.argsort(sims)[::-1][:k]

            # Track retrieval accuracy (same class)
            same_class_count = sum(labels[nn_idx] == labels[q_idx] for nn_idx in top_k)
            key = 'pretrained' if model_idx == 0 else '4way'
            retrieval_stats[key].append(same_class_count / k)

            # Plot query image
            col_offset = model_idx * (k + 1)
            ax = axes[row, col_offset]

            # Load query image
            img, err = load_image_safe(str(img_paths[q_idx]))
            if img is not None:
                ax.imshow(img)
                images_loaded += 1
            else:
                # Show placeholder with error
                ax.set_facecolor('#f0f0f0')
                ax.text(0.5, 0.5, f'{err}',
                        ha='center', va='center', fontsize=7,
                        transform=ax.transAxes, color='#666666')
                images_failed += 1

            ax.set_title(f'Query (C{labels[q_idx]})\n{synsets[q_idx][:10]}', fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')

            # Add model name header on first row
            if row == 0:
                ax.text(0.5, 1.2, name, transform=ax.transAxes,
                        ha='center', fontsize=10, fontweight='bold')

            # Plot k nearest neighbors
            for i, nn_idx in enumerate(top_k):
                ax = axes[row, col_offset + 1 + i]

                # Load neighbor image
                img, err = load_image_safe(str(img_paths[nn_idx]))
                if img is not None:
                    ax.imshow(img)
                    images_loaded += 1
                else:
                    # Show placeholder with error
                    ax.set_facecolor('#f0f0f0')
                    ax.text(0.5, 0.5, f'{err}',
                            ha='center', va='center', fontsize=6,
                            transform=ax.transAxes, color='#666666')
                    images_failed += 1

                # Color border by whether same class
                same_class = labels[nn_idx] == labels[q_idx]
                color = '#2ecc71' if same_class else '#e74c3c'  # Green or red

                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(3)
                    spine.set_visible(True)

                ax.set_title(f'{synsets[nn_idx][:10]}\nsim={sims[nn_idx]:.2f}', fontsize=7)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect('equal')

    # Print retrieval statistics
    print("\n  Retrieval Accuracy (fraction of k neighbors from same class):")
    print(f"    Pretrained: {np.mean(retrieval_stats['pretrained']):.1%}")
    print(f"    4-way:      {np.mean(retrieval_stats['4way']):.1%}")
    print(f"\n  Image loading: {images_loaded} loaded, {images_failed} failed")

    plt.suptitle('Nearest Neighbor Retrieval\n(green = same class, red = different class)',
                 fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {output_path}")
    plt.close()

    return retrieval_stats


def main():
    np.random.seed(SEED)
    ensure_output_dir()

    print("=" * 60)
    print("Nearest Neighbor Retrieval Analysis")
    print("=" * 60)

    feats_list, pca_labels, _, synsets, img_paths, _ = load_data_and_models()

    output_path = os.path.join(OUTPUT_DIR, "nearest_neighbors.png")
    stats = analyze_nearest_neighbors(feats_list, pca_labels, synsets, img_paths, output_path)

    print("\nDone!")
    return stats


if __name__ == "__main__":
    main()
