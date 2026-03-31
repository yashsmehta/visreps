"""Figure 2: coarse-graining visualized in the PCA label space.

Projects 1 image per ImageNet class (1000 points) onto the saved fc2 PCA axes,
then applies median splits to create visually balanced 2-way and 4-way panels.
The 1000-way panel colors each class uniquely.

All panels share the same x/y coordinates — only the coloring changes.

Usage (from project root):
    python manuscript/figures/fig2/pc_scatter_explore.py
    python manuscript/figures/fig2/pc_scatter_explore.py --recompute
"""

import os
import sys
import argparse

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()

from visreps.dataloaders.obj_cls import get_obj_cls_loader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(SCRIPT_DIR, "pc_scatter_1per_class.npz")

EIGVEC_PATH = "datasets/obj_cls/imagenet/eigenvectors_clip.npz"
DATASET = "imagenet-mini-50"

# Color palettes
PALETTES = {
    2: ["#1b9e77", "#d95f02"],
    4: ["#00A896", "#7B68EE", "#E8963E", "#D64045"],
}


# ── Data loading & feature extraction ────────────────────────────────────

def get_dataloader(dataset=DATASET, batch_size=128):
    """Load ImageNet with CLIP preprocessing."""
    import clip
    _, preprocess = clip.load("ViT-L/14", device="cpu")

    data_cfg = {
        "dataset": dataset,
        "batchsize": batch_size,
        "num_workers": 16,
        "data_augment": False,
        "pca_labels_folder": "N/A",
    }
    _, loaders = get_obj_cls_loader(data_cfg, shuffle=False, train_test_split=False)
    loader = loaders["all"]
    loader.dataset.transform = preprocess
    return loader


def extract_clip_features(loader, device):
    """Extract CLIP ViT-L/14 image features, L2-normalized."""
    import clip
    model, _ = clip.load("ViT-L/14", device=device)
    model.eval()

    features = []
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Extracting CLIP", unit="batch"):
            out = model.encode_image(images.to(device))
            out = out / out.norm(dim=-1, keepdim=True)
            features.append(out.float().cpu())

    del model
    torch.cuda.empty_cache()
    return torch.cat(features).numpy()


def compute_and_cache():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load saved eigenvectors
    print(f"Loading eigenvectors from {EIGVEC_PATH}...")
    eigdata = np.load(EIGVEC_PATH)
    eigenvectors = eigdata["eigenvectors"][:, :2]
    mean = eigdata["mean"]
    eigenvalues = eigdata["eigenvalues"][:2]
    total_var = float(eigdata["total_variance"])
    var_explained = eigenvalues / total_var * 100
    print(f"  PC1: {var_explained[0]:.2f}%, PC2: {var_explained[1]:.2f}%")

    # Load dataset and extract features
    loader = get_dataloader()
    imagenet_labels = np.array([s[1] for s in loader.dataset.samples])

    print("\nExtracting CLIP ViT-L/14 features...")
    features = extract_clip_features(loader, device)

    # Project onto saved PCA axes
    pcs_all = (features - mean) @ eigenvectors

    # Select 1 image per class (closest to class centroid for stability)
    print("\nSelecting 1 image per class (1000 total)...")
    selected = []
    rng = np.random.RandomState(42)
    for c in range(1000):
        class_mask = np.where(imagenet_labels == c)[0]
        if len(class_mask) == 0:
            continue
        # Pick the image closest to the class centroid in PC space
        class_pcs = pcs_all[class_mask]
        centroid = class_pcs.mean(axis=0)
        dists = np.linalg.norm(class_pcs - centroid, axis=1)
        best = class_mask[np.argmin(dists)]
        selected.append(best)

    selected = np.array(selected)
    pcs = pcs_all[selected]
    class_labels = imagenet_labels[selected]
    print(f"  Selected {len(selected)} images")

    save_dict = {
        "pcs": pcs,
        "var_explained": var_explained,
        "class_labels": class_labels,
    }
    np.savez_compressed(CACHE_PATH, **save_dict)
    print(f"\nCached -> {CACHE_PATH}")
    return save_dict


# ── Label generation via median splits ───────────────────────────────────

def median_split_labels(pcs, n_way):
    """Assign labels by recursive median splits on PC axes.

    2-way: median split on PC1
    4-way: median on PC1, then median on PC2 within each half
    """
    n = len(pcs)
    labels = np.zeros(n, dtype=int)

    if n_way >= 2:
        med1 = np.median(pcs[:, 0])
        labels[pcs[:, 0] >= med1] = 1

    if n_way >= 4:
        # Split each PC1 half by PC2 median
        new_labels = np.zeros(n, dtype=int)
        for half in [0, 1]:
            mask = labels == half
            med2 = np.median(pcs[mask, 1])
            below = mask & (pcs[:, 1] < med2)
            above = mask & (pcs[:, 1] >= med2)
            new_labels[below] = half * 2
            new_labels[above] = half * 2 + 1
        labels = new_labels

    return labels


# ── Plotting ─────────────────────────────────────────────────────────────

def plot_panel(ax, pcs, labels, n_classes, colors, title,
               point_size=22, alpha=0.75, show_ylabel=True):
    """Draw one scatter panel."""
    rng = np.random.RandomState(42)
    order = rng.permutation(len(labels))

    # Build color array for all points at once
    point_colors = np.array([colors[labels[i] % len(colors)] for i in order])
    ax.scatter(pcs[order, 0], pcs[order, 1],
               c=point_colors, s=point_size, alpha=alpha,
               edgecolors="white", linewidths=0.3,
               rasterized=True, zorder=2)

    ax.set_xlabel("PC 1", fontsize=11, labelpad=5)
    if show_ylabel:
        ax.set_ylabel("PC 2", fontsize=11, labelpad=5)
    else:
        ax.set_ylabel("")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis="both", length=0)

    # Consistent axis limits
    for idx in [0, 1]:
        lo, hi = pcs[:, idx].min(), pcs[:, idx].max()
        margin = (hi - lo) * 0.10
        (ax.set_xlim if idx == 0 else ax.set_ylim)(lo - margin, hi + margin)

    sns.despine(ax=ax, offset=5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()

    if args.recompute or not os.path.exists(CACHE_PATH):
        data = compute_and_cache()
    else:
        print(f"Loading cached data from {CACHE_PATH}")
        raw = np.load(CACHE_PATH, allow_pickle=True)
        data = {k: raw[k] for k in raw.files}

    pcs = data["pcs"]
    class_labels = data["class_labels"]
    var_explained = data["var_explained"]

    # Compute median-split labels on this subset
    labels_2 = median_split_labels(pcs, 2)
    labels_4 = median_split_labels(pcs, 4)

    print(f"2-way split: {np.bincount(labels_2)}")
    print(f"4-way split: {np.bincount(labels_4)}")

    # ── Style ──
    sns.set_theme(style="ticks", context="paper", font_scale=1.1)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
    })

    # ── 3 panels: 2-way, 4-way, 1000-way ──
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8))
    fig.patch.set_facecolor("white")

    # Median values for decision boundary lines
    med_pc1 = np.median(pcs[:, 0])
    med_pc2_left = np.median(pcs[pcs[:, 0] < med_pc1, 1])
    med_pc2_right = np.median(pcs[pcs[:, 0] >= med_pc1, 1])

    # Decision boundary styling
    split_kw = dict(color="#222222", linestyle="--", linewidth=1.3, alpha=0.6, zorder=5)

    # Panel 1: 2-way with PC1 split line
    plot_panel(axes[0], pcs, labels_2, 2, PALETTES[2], "2-way",
               show_ylabel=True)
    axes[0].axvline(med_pc1, **split_kw)

    # Panel 2: 4-way with PC1 + PC2 split lines
    plot_panel(axes[1], pcs, labels_4, 4, PALETTES[4], "4-way",
               show_ylabel=False)
    axes[1].axvline(med_pc1, **split_kw)
    xlim = axes[1].get_xlim()
    axes[1].plot([xlim[0], med_pc1], [med_pc2_left, med_pc2_left],
                 clip_on=True, **split_kw)
    axes[1].plot([med_pc1, xlim[1]], [med_pc2_right, med_pc2_right],
                 clip_on=True, **split_kw)

    # Panel 3: 1000-way — shuffled tab20-based palette for softer colors
    rng_colors = np.random.RandomState(7)
    base_cmap = plt.cm.tab20
    # Generate 1000 colors by cycling tab20 with slight lightness variation
    colors_1k = []
    for i in range(1000):
        base = np.array(base_cmap(i % 20))
        # Add small random variation to avoid exact repeats
        jitter = rng_colors.uniform(-0.08, 0.08, 3)
        base[:3] = np.clip(base[:3] + jitter, 0, 1)
        colors_1k.append(tuple(base))
    # Shuffle so nearby ImageNet classes get different colors
    rng_colors.shuffle(colors_1k)
    plot_panel(axes[2], pcs, class_labels, 1000, colors_1k,
               "1000-way", point_size=20, alpha=0.70, show_ylabel=False)

    plt.tight_layout(w_pad=3.0)
    out = os.path.join(SCRIPT_DIR, "pc_scatter_shared_pca.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white",
                edgecolor="none")
    print(f"\nSaved -> {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
