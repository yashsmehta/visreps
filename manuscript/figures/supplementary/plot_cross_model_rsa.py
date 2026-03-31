"""Supplementary: Cross-model RSA (formerly Figure 2A/B) — Cross-model RSA: coarse ≠ fine-grained.

Computes RSA between internal representations of different models to show:
  (A) Coarse models are fundamentally different from 1000-way (cross-model RSA
      is much lower than inter-seed baseline)
  (B) Low-dimensional projection of 1000-way cannot recover coarse representations

Produces a grouped bar chart with three comparison types per granularity level.

Usage (from project root):
    # Compute RSMs and plot (first run — takes ~10-15 min on 4090)
    python manuscript/figures/fig3/plot_cross_model_rsa.py

    # Re-plot from cached data
    python manuscript/figures/fig3/plot_cross_model_rsa.py --plot-only

    # Customize
    python manuscript/figures/fig3/plot_cross_model_rsa.py --layer fc1 --n_images 1500
"""

import os
import sys
import json
import argparse

import numpy as np
import torch
import scipy.stats
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
import seaborn as sns

sys.path.insert(0, ".")
sys.path.insert(0, "experiments/coarse_grain_benefits")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(SCRIPT_DIR, "cross_model_rsa_data.json")

# ── Config ───────────────────────────────────────────────────────────────────
COARSE_CFGS = [2, 4, 8, 16, 32, 64]
CHECKPOINT_DIR = "/data/ymehta3/alexnet_pca"
CHECKPOINT_DIR_1K = "/data/ymehta3/default"
# ── Style ────────────────────────────────────────────────────────────────────
sns.set_theme(style="ticks", context="paper", font_scale=1.05)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.linewidth": 1.0,
})

COLOR_INTERSEED = "#AAAAAA"     # Light gray — inter-seed 1K baseline
COLOR_CROSS = "#2166AC"          # Blue — 1K vs coarse
COLOR_PROJECTED = "#E08214"      # Orange — projected-1K vs coarse


def seed_letter(seed):
    return chr(ord("a") + seed - 1)


def load_model(cfg_id, seed, device):
    """Load a trained model checkpoint."""
    from utils import load_model_by_config
    if cfg_id == 1000:
        return load_model_by_config(cfg_id, seed, checkpoint_dir=CHECKPOINT_DIR_1K,
                                    device=device)
    return load_model_by_config(cfg_id, seed, checkpoint_dir=CHECKPOINT_DIR,
                                device=device)


def get_dataloader(n_images=1000, batch_size=128):
    """Get a dataloader with a random subset of ImageNet validation images."""
    from torchvision import transforms, datasets

    imagenet_dir = os.environ.get("IMAGENET_DATA_DIR", "")
    if not imagenet_dir or not os.path.isdir(imagenet_dir):
        raise RuntimeError(
            "IMAGENET_DATA_DIR not set or invalid. Source .env first.")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(imagenet_dir, transform=transform)

    # Deterministic subsample
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dataset), size=min(n_images, len(dataset)),
                         replace=False)
    subset = torch.utils.data.Subset(dataset, indices)
    return torch.utils.data.DataLoader(subset, batch_size=batch_size,
                                       shuffle=False, num_workers=4,
                                       pin_memory=True)


@torch.no_grad()
def extract_features(model, loader, layer, device):
    """Extract features from one layer using FeatureExtractor."""
    from utils import get_feature_extractor
    fe = get_feature_extractor(model, [layer])
    model.eval()

    all_feats = []
    for images, _ in loader:
        images = images.to(device)
        out = fe(images)
        feat = out[layer]
        if feat.ndim > 2:
            feat = feat.flatten(1)
        all_feats.append(feat.cpu().numpy())

    return np.concatenate(all_feats, axis=0)


def compute_rsm(features):
    """Compute Pearson correlation RSM (n_images x n_images)."""
    # Center features
    features = features - features.mean(axis=1, keepdims=True)
    # Normalize
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1
    features = features / norms
    # Pearson correlation = dot product of centered, normalized features
    rsm = features @ features.T
    return rsm


def rsm_correlation(rsm1, rsm2, method="spearman"):
    """Compare two RSMs using upper-triangular elements."""
    n = rsm1.shape[0]
    idx = np.triu_indices(n, k=1)
    v1 = rsm1[idx]
    v2 = rsm2[idx]

    if method == "spearman":
        r, _ = scipy.stats.spearmanr(v1, v2)
    elif method == "kendall":
        r, _ = scipy.stats.kendalltau(v1, v2)
    else:
        r, _ = scipy.stats.pearsonr(v1, v2)
    return float(r)


def project_to_top_pcs(features, k):
    """Project features onto their top-k principal components."""
    centered = features - features.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    # Reconstruct from top-k PCs
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]


def compute_all(layer, n_images, compare_method):
    """Compute all cross-model RSA comparisons. Returns dict of results."""
    from dotenv import load_dotenv
    load_dotenv()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Layer: {layer}, N images: {n_images}, Method: {compare_method}")

    loader = get_dataloader(n_images=n_images)
    print(f"Loaded {len(loader.dataset)} images")

    results = {"layer": layer, "n_images": n_images, "method": compare_method,
               "comparisons": {}}

    # 1. Extract 1000-way features (two seeds for inter-seed baseline)
    print("\n--- Extracting 1000-way features ---")
    feats_1k = {}
    for seed in [1, 2]:
        print(f"  Loading 1000-way seed {seed}...")
        model = load_model(1000, seed, device)
        feats_1k[seed] = extract_features(model, loader, layer, device)
        del model
        torch.cuda.empty_cache()

    rsm_1k_s1 = compute_rsm(feats_1k[1])
    rsm_1k_s2 = compute_rsm(feats_1k[2])
    interseed_1k = rsm_correlation(rsm_1k_s1, rsm_1k_s2, compare_method)
    print(f"  Inter-seed 1K RSA: {interseed_1k:.4f}")
    results["interseed_1k"] = interseed_1k

    # Precompute SVD of 1K features (seed 1) for projection — reused across cfg_ids
    centered_1k = feats_1k[1] - feats_1k[1].mean(axis=0, keepdims=True)
    U_1k, S_1k, Vt_1k = np.linalg.svd(centered_1k, full_matrices=False)

    # 2. For each coarseness level, compare with 1000-way
    for cfg_id in COARSE_CFGS:
        print(f"\n--- cfg_id={cfg_id} ---")
        comp = {}

        # Extract coarse features (seed 1)
        try:
            model = load_model(cfg_id, 1, device)
        except Exception as e:
            print(f"  Could not load cfg_id={cfg_id} seed=1: {e}")
            results["comparisons"][str(cfg_id)] = {"error": str(e)}
            continue
        feats_coarse = extract_features(model, loader, layer, device)
        del model
        torch.cuda.empty_cache()

        rsm_coarse = compute_rsm(feats_coarse)

        # A: 1K (seed 1) vs coarse (seed 1)
        cross = rsm_correlation(rsm_1k_s1, rsm_coarse, compare_method)
        comp["cross_1k_coarse"] = cross
        print(f"  1K vs {cfg_id}-way RSA: {cross:.4f}")

        # B: Projected 1K (top-k PCs) vs coarse — reuse precomputed SVD
        n_pcs = max(int(np.log2(cfg_id)), 1)
        feats_1k_projected = U_1k[:, :n_pcs] @ np.diag(S_1k[:n_pcs]) @ Vt_1k[:n_pcs, :]
        rsm_1k_proj = compute_rsm(feats_1k_projected)
        proj = rsm_correlation(rsm_1k_proj, rsm_coarse, compare_method)
        comp["projected_1k_coarse"] = proj
        comp["n_pcs_used"] = n_pcs
        print(f"  Projected-1K (top-{n_pcs} PCs) vs {cfg_id}-way RSA: {proj:.4f}")

        # Inter-seed coarse (if seed 2 exists)
        try:
            model2 = load_model(cfg_id, 2, device)
            feats_coarse_s2 = extract_features(model2, loader, layer, device)
            del model2
            torch.cuda.empty_cache()
            rsm_coarse_s2 = compute_rsm(feats_coarse_s2)
            interseed_coarse = rsm_correlation(rsm_coarse, rsm_coarse_s2,
                                               compare_method)
            comp["interseed_coarse"] = interseed_coarse
            print(f"  Inter-seed {cfg_id}-way RSA: {interseed_coarse:.4f}")
        except Exception as e:
            print(f"  Could not load cfg_id={cfg_id} seed=2: {e}")

        results["comparisons"][str(cfg_id)] = comp

    # Save cache
    with open(CACHE_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nCached results -> {CACHE_PATH}")

    return results


def plot_results(results):
    """Plot projected-1K vs coarse RSA (Panel B of Figure 2).

    Shows single bars per granularity level: how well a low-rank projection
    of the 1K model (onto k = log2(n_classes) PCs) matches the coarse model.
    Inter-seed 1K baseline shown as dashed line for reference.
    """
    comparisons = results["comparisons"]
    method = results.get("method", "spearman")
    layer = results.get("layer", "fc1")

    valid_cfgs = [c for c in COARSE_CFGS if str(c) in comparisons
                  and "error" not in comparisons[str(c)]]

    if not valid_cfgs:
        print("No valid data to plot")
        return

    n = len(valid_cfgs)
    x = np.arange(n)
    bar_width = 0.45

    fig, ax = plt.subplots(figsize=(max(5.5, n * 0.95), 3.8))

    # Inter-seed 1K baseline (horizontal dashed line)
    interseed_1k = results.get("interseed_1k", np.nan)
    if not np.isnan(interseed_1k):
        ax.axhline(interseed_1k, color=COLOR_INTERSEED, linestyle="--",
                    linewidth=1.5, label=f"Inter-seed 1K: {interseed_1k:.2f}",
                    zorder=1)

    # Single bars: projected-1K vs coarse
    proj_vals = []
    n_pcs_labels = []
    for cfg_id in valid_cfgs:
        comp = comparisons[str(cfg_id)]
        proj_vals.append(comp.get("projected_1k_coarse", np.nan))
        n_pcs_labels.append(comp.get("n_pcs_used", int(np.log2(cfg_id))))

    bars = ax.bar(x, proj_vals, bar_width,
                  color=COLOR_PROJECTED, edgecolor="#333333", linewidth=0.6,
                  label="Projected-1K vs. Coarse", zorder=3)

    # Annotate number of PCs above each bar
    for i, (val, k) in enumerate(zip(proj_vals, n_pcs_labels)):
        if not np.isnan(val):
            ax.text(i, val + 0.015, f"k={k}", ha="center", va="bottom",
                    fontsize=7.5, color="#555555", fontstyle="italic")

    # Axis formatting
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in valid_cfgs], fontsize=10)
    ax.set_xlabel("Number of Classes", fontsize=10, labelpad=6)
    method_label = method.capitalize()
    ax.set_ylabel(f"RSA ({method_label} " + r"$\rho$)", fontsize=10, labelpad=6)

    ax.set_ylim(0, max(interseed_1k + 0.1, 0.9) if not np.isnan(interseed_1k) else 0.5)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="both", labelsize=9, length=4, width=0.8)
    ax.yaxis.grid(True, which="major", color="#EBEBEB", linewidth=0.4, zorder=0)

    ax.legend(loc="upper left", fontsize=8, frameon=True,
              edgecolor="#cccccc", fancybox=False, handletextpad=0.4)

    sns.despine(ax=ax, offset=5)
    plt.tight_layout()

    out = os.path.join(SCRIPT_DIR, f"cross_model_rsa_{layer}.png")
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white",
                edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", default="fc1",
                        choices=["conv1", "conv2", "conv3", "conv4", "conv5",
                                 "fc1", "fc2"])
    parser.add_argument("--n_images", type=int, default=1000,
                        help="Number of ImageNet images for RSM computation")
    parser.add_argument("--method", default="spearman",
                        choices=["spearman", "kendall", "pearson"])
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip computation, re-plot from cached data")
    args = parser.parse_args()

    if args.plot_only:
        if not os.path.exists(CACHE_PATH):
            print(f"No cached data at {CACHE_PATH}. Run without --plot-only first.")
            return
        with open(CACHE_PATH) as f:
            results = json.load(f)
    else:
        results = compute_all(args.layer, args.n_images, args.method)

    plot_results(results)


if __name__ == "__main__":
    main()
