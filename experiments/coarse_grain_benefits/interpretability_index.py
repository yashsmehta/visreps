"""
Class Selectivity Index (CSI): Measures per-neuron class selectivity.

Compares interpretability of coarse-grained vs fine-grained trained models.

    CSI(neuron) = (mu_max - mu_other) / (mu_max + mu_other)
        mu_max:   mean activation for the most-preferred class
        mu_other: mean activation across all other classes

All models are evaluated using ImageNet-1K class labels as the common
reference taxonomy, regardless of training granularity. This avoids the
circularity of measuring selectivity w.r.t. training labels.

Reference: Morcos et al. (2018) "On the importance of single directions"

Run from repo root:
    python experiments/coarse_grain_benefits/interpretability_index.py
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import (
    get_device, ensure_output_dir, load_model_by_config,
    get_model_configs, get_config_name,
)
from visreps.models.utils import FeatureExtractor
from visreps.dataloaders.obj_cls import get_obj_cls_loader


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
LAYERS = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2"]
BATCH_SIZE = 256
NUM_WORKERS = 4
N_CLASSES_REF = 1000  # Always use ImageNet-1K labels

# Plot style
COLORS = {32: "#E69F00", 64: "#2E8B57", 1000: "#0072B2", "pretrained": "#999999"}
MARKERS = {32: "s", 64: "^", 1000: "o", "pretrained": "D"}


# ─────────────────────────────────────────────────────────────
# CSI COMPUTATION
# ─────────────────────────────────────────────────────────────
def compute_csi(class_means):
    """
    Compute Class Selectivity Index per neuron.

    Args:
        class_means: (n_classes, n_neurons) mean activation per class

    Returns:
        csi:       (n_neurons,) selectivity index
        preferred: (n_neurons,) index of most-preferred class
    """
    preferred = np.argmax(class_means, axis=0)
    n_neurons = class_means.shape[1]
    n_classes = class_means.shape[0]

    mu_max = class_means[preferred, np.arange(n_neurons)]

    # Mean of all other class means
    total = class_means.sum(axis=0)
    mu_other = (total - mu_max) / max(n_classes - 1, 1)

    num = mu_max - mu_other
    den = mu_max + mu_other
    csi = np.where(np.abs(den) < 1e-10, 0.0, num / den)

    return csi, preferred


def compute_csi_for_model(model, loader, layers, device):
    """
    Compute layer-wise CSI for a model using online accumulation.

    Applies Global Average Pooling to conv layers so each filter = one neuron.
    Accumulates per-class sums and counts without storing all activations.
    """
    return_nodes = {l: l for l in layers}
    extractor = FeatureExtractor(model, return_nodes=return_nodes)
    extractor = extractor.to(device).eval()

    class_sums = {}
    class_counts = np.zeros(N_CLASSES_REF, dtype=np.int64)

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Accumulating activations"):
            images = images.to(device)
            features = extractor(images)
            labels_np = labels.numpy()

            # Update class counts
            class_counts += np.bincount(labels_np, minlength=N_CLASSES_REF)

            for layer in layers:
                feat = features[layer]

                # GAP for conv layers: (B, C, H, W) -> (B, C)
                if feat.dim() == 4:
                    feat = feat.mean(dim=(2, 3))
                elif feat.dim() > 2:
                    feat = feat.view(feat.size(0), -1)

                feat_np = feat.cpu().float().numpy()

                if layer not in class_sums:
                    class_sums[layer] = np.zeros(
                        (N_CLASSES_REF, feat_np.shape[1]), dtype=np.float64
                    )

                np.add.at(class_sums[layer], labels_np, feat_np)

    # Compute CSI per layer
    valid = class_counts > 0
    results = {}
    for layer in layers:
        means = np.zeros_like(class_sums[layer])
        means[valid] = class_sums[layer][valid] / class_counts[valid, None]
        means_valid = means[valid]

        csi, preferred = compute_csi(means_valid)
        results[layer] = {
            "median_csi": float(np.median(csi)),
            "mean_csi": float(np.mean(csi)),
            "std_csi": float(np.std(csi)),
            "n_neurons": len(csi),
            "n_classes_with_samples": int(valid.sum()),
        }

    return results


# ─────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────
def plot_results(df, output_path):
    """Layer-wise CSI profile, averaged across seeds with SEM error bands."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 300,
    })

    fig, ax = plt.subplots(figsize=(5.5, 3.5))

    cfg_ids = sorted(
        df["cfg_id"].unique(),
        key=lambda x: int(x) if str(x).isdigit() else 9999,
    )

    for cfg_id in cfg_ids:
        sub = df[df["cfg_id"] == cfg_id]
        grouped = sub.groupby("depth_normalized")["median_csi"]
        depths = grouped.mean().index.values
        means = grouped.mean().values
        n_seeds = grouped.count().values
        sems = grouped.std().values / np.sqrt(np.maximum(n_seeds, 1))

        color = COLORS.get(cfg_id, "#333333")
        marker = MARKERS.get(cfg_id, "o")
        label = f"{cfg_id}-way" if isinstance(cfg_id, int) else str(cfg_id)

        ax.plot(
            depths, means,
            color=color, marker=marker, markersize=5,
            markerfacecolor=color, markeredgecolor="white",
            markeredgewidth=0.4, linewidth=1.5, label=label, zorder=3,
        )
        ax.fill_between(
            depths, means - sems, means + sems,
            color=color, alpha=0.15, zorder=2,
        )

    # Add layer labels on x-axis
    n_layers = len(LAYERS)
    tick_positions = [i / (n_layers - 1) for i in range(n_layers)]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(LAYERS, rotation=45, ha="right")

    ax.set_xlabel("Layer")
    ax.set_ylabel("Median CSI")
    ax.set_title("Class Selectivity Index")
    ax.set_xlim(-0.05, 1.05)
    ax.legend(frameon=True, framealpha=0.9, edgecolor="none")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {output_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    device = get_device()
    print(f"Device: {device}")
    print(f"Reference taxonomy: ImageNet-{N_CLASSES_REF}")
    print(f"Layers: {LAYERS}\n")

    # Load ImageNet with original 1K labels
    cfg_loader = {
        "dataset": "imagenet",
        "batchsize": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pca_labels": False,
    }
    _, loaders = get_obj_cls_loader(
        cfg_loader, shuffle=False, preprocess=True, train_test_split=False
    )
    loader = loaders["all"]
    print(f"Dataset: {len(loader.dataset)} images\n")

    # Evaluate models
    configs = get_model_configs(cfg_ids=[32, 64, 1000], seeds=[1, 2, 3])

    all_results = []
    n_layers = len(LAYERS)

    for cfg_id, seed in configs:
        name = get_config_name(cfg_id, seed)
        print(f"\n{'='*50}")
        print(f"Model: {name}")

        try:
            model = load_model_by_config(cfg_id, seed, device=device)
        except FileNotFoundError as e:
            print(f"  Skipping: {e}")
            continue

        results = compute_csi_for_model(model, loader, LAYERS, device)

        for i, layer in enumerate(LAYERS):
            r = results[layer]
            print(
                f"  {layer:6s}: median={r['median_csi']:.4f}  "
                f"mean={r['mean_csi']:.4f}  n_neurons={r['n_neurons']}"
            )
            all_results.append({
                "cfg_id": cfg_id,
                "seed": seed,
                "model_name": name,
                "layer": layer,
                "depth_normalized": i / (n_layers - 1),
                **r,
            })

        del model
        torch.cuda.empty_cache()

    # Save results
    output_dir = ensure_output_dir()
    csv_path = os.path.join(output_dir, "class_selectivity_index.csv")
    plot_path = os.path.join(output_dir, "class_selectivity_index.png")

    df = pd.DataFrame(all_results)
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Plot
    plot_results(df, plot_path)


if __name__ == "__main__":
    main()
