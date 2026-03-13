"""Figure 2A — Class-level RDM grid: 1000-way + coarse models (2,4,8,16,32).

Extracts fc1 activations from ImageNet, averages per class (10 images each),
computes 1000×1000 Pearson RDMs for each granularity level.

Usage (from project root):
    # Compute features + save cache (~5-10 min on 4090)
    python manuscript/figures/fig2/plot_class_rdms.py

    # Re-plot from cached data
    python manuscript/figures/fig2/plot_class_rdms.py --plot-only
"""

import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/coarse_grain_benefits")

from visreps.analysis.rsa import compute_rdm, compute_rdm_correlation

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(SCRIPT_DIR, "class_rdm_data.npz")

# ── Config ────────────────────────────────────────────────────────────────
LAYER = "fc1"
N_IMAGES_PER_CLASS = 10
CHECKPOINT_DIR_1K = "/data/ymehta3/default"
CHECKPOINT_DIR_CLIP = "/data/ymehta3/clip_pca"
COARSE_CFG_IDS = [4, 8, 16, 32, 64]
SEED = 1

SEMANTIC_LABELS_PATH = "experiments/wordnet/semantic_categories.csv"
SEMANTIC_MAPPING_PATH = "experiments/wordnet/semantic_categories_mapping.txt"

CATEGORY_NAMES = [
    "Animals", "Natural World", "Food & Produce",
    "Structures & Architecture", "Domestic & Apparel",
    "Vehicles & Transport", "Tools & Electronics", "General Objects",
]

# Saturated palette for 8 categories (chosen for contrast on dark background)
CATEGORY_COLORS = [
    "#e31a1c",  # Animals — red
    "#33a02c",  # Natural World — green
    "#ff7f00",  # Food & Produce — orange
    "#1f78b4",  # Structures — blue
    "#6a3d9a",  # Domestic — purple
    "#b15928",  # Vehicles — brown
    "#e377c2",  # Tools & Electronics — pink
    "#999999",  # General Objects — gray
]

sns.set_theme(style="ticks", context="paper", font_scale=1.0)


# ── Data loading ──────────────────────────────────────────────────────────

def _load_imagenet_dataset():
    """Load ImageNet dataset with standard eval transforms."""
    from dotenv import load_dotenv
    load_dotenv()
    import visreps.utils as vutils
    from visreps.dataloaders.obj_cls import ImageNetDataset, get_transform

    base_path = vutils.get_env_var("IMAGENET_DATA_DIR")
    transform = get_transform(ds_stats="imgnet", data_augment=False)
    return ImageNetDataset(base_path, split="all", transform=transform)


def get_class_to_category(dataset):
    """Map ImageNet class_idx (0-999) → super-category label (0-7).

    Uses the semantic_categories.csv which maps image_id → pca_label (0-7).
    Since all images of the same class share the same category, we just
    need one image per class.
    """
    # Load semantic labels
    sem_df = pd.read_csv(SEMANTIC_LABELS_PATH)
    sem_map = dict(zip(sem_df["image"], sem_df["pca_label"]))

    # Build class → category mapping
    class_to_cat = {}
    for img_path, class_idx, img_id in dataset.samples:
        if class_idx not in class_to_cat and img_id in sem_map:
            class_to_cat[class_idx] = sem_map[img_id]
        if len(class_to_cat) == 1000:
            break

    return class_to_cat


def get_class_image_indices(dataset, n_per_class=10, seed=42):
    """Get indices of n_per_class images for each of the 1000 classes.

    Returns dict: class_idx → list of dataset indices.
    """
    rng = np.random.default_rng(seed)
    class_indices = {}
    for idx, (_, class_idx, _) in enumerate(dataset.samples):
        if class_idx not in class_indices:
            class_indices[class_idx] = []
        class_indices[class_idx].append(idx)

    # Subsample
    result = {}
    for class_idx, indices in class_indices.items():
        if len(indices) >= n_per_class:
            result[class_idx] = rng.choice(indices, size=n_per_class,
                                            replace=False).tolist()
        else:
            result[class_idx] = indices
    return result


@torch.no_grad()
def extract_class_centroids(model, dataset, class_image_indices, layer, device):
    """Extract fc1 features, average per class → (1000, n_features) matrix."""
    from utils import get_feature_extractor

    fe = get_feature_extractor(model, [layer])
    model.eval()

    # Collect all indices we need
    all_indices = []
    index_to_class = {}
    for class_idx, indices in class_image_indices.items():
        for idx in indices:
            all_indices.append(idx)
            index_to_class[idx] = class_idx

    # Build a subset dataloader
    subset = torch.utils.data.Subset(dataset, all_indices)
    loader = torch.utils.data.DataLoader(subset, batch_size=128,
                                          shuffle=False, num_workers=4,
                                          pin_memory=True)

    # Extract features
    all_feats = []
    for images, _ in loader:
        images = images.to(device)
        out = fe(images)
        feat = out[layer]
        if feat.ndim > 2:
            feat = feat.flatten(1)
        all_feats.append(feat.cpu())
    all_feats = torch.cat(all_feats, dim=0)  # (n_total_images, n_features)

    # Average per class
    n_classes = 1000
    n_features = all_feats.shape[1]
    centroids = torch.zeros(n_classes, n_features)
    counts = torch.zeros(n_classes)

    for i, idx in enumerate(all_indices):
        class_idx = index_to_class[idx]
        centroids[class_idx] += all_feats[i]
        counts[class_idx] += 1

    # Avoid division by zero
    valid = counts > 0
    centroids[valid] /= counts[valid].unsqueeze(1)

    return centroids.numpy(), valid.numpy()


def compute_data(layer, n_per_class):
    """Compute class centroids for 1000-way + all coarse models, save to cache."""
    from utils import load_model_by_config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print("Loading ImageNet dataset...")
    dataset = _load_imagenet_dataset()
    print(f"  {len(dataset)} images")

    # Get per-class image indices
    print(f"Selecting {n_per_class} images per class...")
    class_image_indices = get_class_image_indices(dataset, n_per_class)

    # Get class → category mapping
    print("Loading WordNet category mapping...")
    class_to_cat = get_class_to_category(dataset)
    categories = np.array([class_to_cat.get(i, -1) for i in range(1000)])

    # Extract centroids: 1000-way model
    print(f"\n--- 1000-way model (seed {SEED}) ---")
    model_1k = load_model_by_config(1000, SEED, checkpoint_dir=CHECKPOINT_DIR_1K,
                                     device=device)
    centroids_1k, valid_1k = extract_class_centroids(
        model_1k, dataset, class_image_indices, layer, device)
    del model_1k
    torch.cuda.empty_cache()
    print(f"  Extracted centroids: {centroids_1k.shape}")

    # Extract centroids: each coarse model
    coarse_centroids = {}
    for cfg_id in COARSE_CFG_IDS:
        print(f"\n--- {cfg_id}-way CLIP-PCA model (seed {SEED}) ---")
        model = load_model_by_config(cfg_id, SEED,
                                     checkpoint_dir=CHECKPOINT_DIR_CLIP,
                                     device=device)
        centroids, _ = extract_class_centroids(
            model, dataset, class_image_indices, layer, device)
        coarse_centroids[cfg_id] = centroids
        del model
        torch.cuda.empty_cache()
        print(f"  Extracted centroids: {centroids.shape}")

    # Save cache — store coarse centroids keyed by cfg_id
    save_dict = {
        "centroids_1k": centroids_1k,
        "categories": categories,
        "valid_1k": valid_1k,
    }
    for cfg_id, cent in coarse_centroids.items():
        save_dict[f"centroids_{cfg_id}"] = cent
    np.savez(CACHE_PATH, **save_dict)
    print(f"\nCached -> {CACHE_PATH}")

    return centroids_1k, coarse_centroids, categories


# ── Plotting ──────────────────────────────────────────────────────────────

def rank_transform(rdm):
    """Rank upper triangle, mirror to lower, scale to [0, 1]."""
    n = rdm.shape[0]
    triu = np.triu_indices(n, k=1)
    ranks = rankdata(rdm[triu]) / rdm[triu].size
    ranked = np.zeros_like(rdm)
    ranked[triu] = ranks
    ranked.T[triu] = ranks
    return ranked


def build_sort_order(categories, rdm):
    """Sort classes by category, then hierarchical clustering within each block."""
    unique_cats = sorted(set(categories))
    sorted_indices = []
    block_boundaries = []
    offset = 0

    for cat in unique_cats:
        member_idx = np.where(categories == cat)[0]
        if len(member_idx) <= 2:
            order = member_idx
        else:
            sub_rdm = rdm[np.ix_(member_idx, member_idx)]
            sub_condensed = squareform(sub_rdm, checks=False)
            # Clamp any negatives from numerical noise
            sub_condensed = np.maximum(sub_condensed, 0)
            sub_order = leaves_list(linkage(sub_condensed, method="average"))
            order = member_idx[sub_order]

        cat_name = CATEGORY_NAMES[cat] if cat < len(CATEGORY_NAMES) else f"Cat {cat}"
        block_boundaries.append((offset, cat, len(order), cat_name))
        sorted_indices.extend(order)
        offset += len(order)

    return np.array(sorted_indices), block_boundaries


def draw_sidebar(ax, block_boundaries, n, side="left",
                 width_frac=0.018, gap_frac=0.005):
    """Draw colored sidebar along RDM edge."""
    w = n * width_frac
    gap = n * gap_frac
    for start, cat, size, _ in block_boundaries:
        color = CATEGORY_COLORS[cat] if cat < len(CATEGORY_COLORS) else "#888888"
        if side == "left":
            rect = mpatches.Rectangle(
                (-w - gap, start - 0.5), w, size,
                facecolor=color, edgecolor="none", clip_on=False)
        else:
            rect = mpatches.Rectangle(
                (start - 0.5, n - 0.5 + gap), size, w,
                facecolor=color, edgecolor="none", clip_on=False)
        ax.add_patch(rect)


def draw_boundaries(ax, block_boundaries, n, color="white", lw=0.3, alpha=0.5):
    """Draw thin lines at category boundaries."""
    for start, _, size, _ in block_boundaries:
        if start > 0:
            ax.axhline(start - 0.5, color=color, lw=lw, alpha=alpha)
            ax.axvline(start - 0.5, color=color, lw=lw, alpha=alpha)


def plot_rdm_panel(ax, rdm, block_boundaries, n, title, rsa_score=None):
    """Draw a single RDM panel with category annotations (raw dissimilarity)."""
    im = ax.imshow(rdm, cmap="magma", interpolation="nearest",
                   aspect="equal", rasterized=True, vmin=0, vmax=2)

    ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
    if rsa_score is not None:
        ax.text(0.5, 1.01, f"Cross-model $\\rho_s$ = {rsa_score:.3f}",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=7, color="#555555")

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    draw_boundaries(ax, block_boundaries, n)
    draw_sidebar(ax, block_boundaries, n, side="left")
    draw_sidebar(ax, block_boundaries, n, side="bottom")

    return im


def plot_figure(centroids_1k, coarse_centroids, categories):
    """Create the 2x3 RDM grid figure (standalone, not used by figure2.py)."""
    valid = categories >= 0
    centroids_1k = centroids_1k[valid]
    categories = categories[valid]
    n_classes = len(categories)
    print(f"Plotting RDMs for {n_classes} valid classes")

    # Compute 1000-way RDM
    rdm_1k = compute_rdm(torch.tensor(centroids_1k, dtype=torch.float32)).numpy()
    sort_idx, block_boundaries = build_sort_order(categories, rdm_1k)
    rdm_1k_sorted = rdm_1k[np.ix_(sort_idx, sort_idx)]

    # Compute coarse RDMs
    coarse_rdms = {}
    for cfg_id, cent in coarse_centroids.items():
        cent_valid = cent[valid]
        rdm = compute_rdm(torch.tensor(cent_valid, dtype=torch.float32)).numpy()
        coarse_rdms[cfg_id] = rdm[np.ix_(sort_idx, sort_idx)]

    # ── Plot 2×3 grid ──
    all_cfg_ids = [1000] + COARSE_CFG_IDS
    fig, axes = plt.subplots(2, 3, figsize=(12, 8.5))
    axes = axes.flatten()

    for i, cfg_id in enumerate(all_cfg_ids):
        ax = axes[i]
        rdm = rdm_1k_sorted if cfg_id == 1000 else coarse_rdms[cfg_id]
        title = "1000-way" if cfg_id == 1000 else f"{cfg_id}-way"
        im = plot_rdm_panel(ax, rdm, block_boundaries, n_classes, title)

    # Hide 6th subplot if only 5 coarse + 1000 = 6
    if len(all_cfg_ids) < 6:
        axes[len(all_cfg_ids)].axis("off")

    fig.subplots_adjust(right=0.92, wspace=0.08, hspace=0.15)
    cax = fig.add_axes([0.93, 0.25, 0.015, 0.5])
    cb = fig.colorbar(im, cax=cax)
    cb.ax.tick_params(labelsize=7, length=2, width=0.4)
    cb.outline.set_linewidth(0.4)
    cb.set_label("Pearson dissimilarity (1 - r)", fontsize=7, labelpad=4)

    out = os.path.join(SCRIPT_DIR, "class_rdm_comparison.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white",
                edgecolor="none")
    plt.close()
    print(f"Saved -> {out}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip computation, re-plot from cached data")
    parser.add_argument("--n_per_class", type=int, default=N_IMAGES_PER_CLASS)
    args = parser.parse_args()

    if args.plot_only:
        if not os.path.exists(CACHE_PATH):
            print(f"No cached data at {CACHE_PATH}. Run without --plot-only first.")
            return
        print(f"Loading cached data from {CACHE_PATH}")
        data = np.load(CACHE_PATH)
        centroids_1k = data["centroids_1k"]
        categories = data["categories"]
        coarse_centroids = {}
        for cfg_id in COARSE_CFG_IDS:
            key = f"centroids_{cfg_id}"
            if key in data:
                coarse_centroids[cfg_id] = data[key]
    else:
        centroids_1k, coarse_centroids, categories = compute_data(
            LAYER, args.n_per_class)

    plot_figure(centroids_1k, coarse_centroids, categories)


if __name__ == "__main__":
    main()
