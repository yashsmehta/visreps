"""
Class Selectivity Index (CSI): Measures per-neuron class selectivity.

Compares two 1000-way classifiers:
    1. Direct 1000-way training
    2. Curriculum: 64-way pretrained -> fine-tuned on 1000-way (late_layers)

    CSI(neuron) = (mu_max - mu_other) / (mu_max + mu_other)
        mu_max:   mean activation for the most-preferred class
        mu_other: mean activation across all other classes

Outputs:
    - Per-layer mean CSI table
    - Histogram (PDF) of CSI pooled across all layers for both models

Reference: Morcos et al. (2018) "On the importance of single directions"

Run from repo root:
    python experiments/coarse_grain_benefits/class_selectivity_index.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from visreps.models.utils import FeatureExtractor
from visreps.dataloaders.obj_cls import get_obj_cls_loader


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
MODELS = {
    "Direct 1000-way": {
        "path": "/data/ymehta3/default/cfg1000a/checkpoint_epoch_20.pth",
    },
    "Curriculum (64→1000)": {
        "path": os.path.join(
            PROJECT_ROOT,
            "experiments/coarse_grain_benefits/results/curriculum_checkpoints",
            "cfg64_to_1000_late_layers_a/checkpoint_epoch_10.pth",
        ),
    },
}

LAYERS = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2"]
BATCH_SIZE = 256
NUM_WORKERS = 4
N_CLASSES = 1000

# Validation mode: use test split for fast pipeline check
VALIDATION_MODE = False

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "experiments/coarse_grain_benefits/results")

# Plot style
COLORS = {
    "Direct 1000-way": "#0072B2",
    "Curriculum (64→1000)": "#E69F00",
}

PLOT_STYLE = {
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
}


# ─────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────
def load_model(path, device):
    """Load a trained checkpoint from path."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return checkpoint['model'].to(device).eval()


# ─────────────────────────────────────────────────────────────
# CSI COMPUTATION
# ─────────────────────────────────────────────────────────────
def compute_csi(class_means):
    """
    Compute Class Selectivity Index per neuron.

    Args:
        class_means: (n_classes, n_neurons) mean activation per class

    Returns:
        csi: (n_neurons,) selectivity index in [0, 1]
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

    return csi


def accumulate_activations(model, loader, layers, device):
    """
    Accumulate per-class (1K) activation sums and counts in a single pass.

    Returns:
        class_sums: dict[layer] -> (1000, n_neurons) sum of activations per class
        class_counts: (1000,) number of images per class
    """
    return_nodes = {l: l for l in layers}
    extractor = FeatureExtractor(model, return_nodes=return_nodes)
    extractor = extractor.to(device).eval()

    class_sums = {}
    class_counts = np.zeros(N_CLASSES, dtype=np.int64)

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Accumulating activations"):
            images = images.to(device)
            features = extractor(images)
            labels_np = labels.numpy()

            class_counts += np.bincount(labels_np, minlength=N_CLASSES)

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
                        (N_CLASSES, feat_np.shape[1]), dtype=np.float64
                    )
                np.add.at(class_sums[layer], labels_np, feat_np)

    n_with_samples = (class_counts > 0).sum()
    print(f"  {class_counts.sum()} images, "
          f"{n_with_samples}/{N_CLASSES} classes with samples")

    return class_sums, class_counts


def compute_layer_csi(class_sums, class_counts, layer):
    """
    Compute per-neuron CSI for a layer at 1000-class reference.

    Returns:
        csi: (n_neurons,) raw CSI values
    """
    sums = class_sums[layer]
    valid = class_counts > 0
    means = np.zeros_like(sums)
    means[valid] = sums[valid] / class_counts[valid, None]
    means = means[valid]
    return compute_csi(means)


# ─────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────
def plot_csi_histogram(csi_per_model, output_path):
    """
    Overlapping histograms of CSI pooled across all layers for each model.
    """
    plt.rcParams.update(PLOT_STYLE)

    fig, ax = plt.subplots(figsize=(5, 3.5))

    bins = np.linspace(0, 1, 51)

    for model_name, csi_values in csi_per_model.items():
        color = COLORS.get(model_name, "#333333")
        ax.hist(
            csi_values, bins=bins, density=True,
            color=color, alpha=0.5, label=model_name,
            edgecolor="white", linewidth=0.3,
        )

    ax.set_xlabel("Class Selectivity Index")
    ax.set_ylabel("Density")
    ax.set_title("CSI Distribution (all layers)")
    ax.legend(frameon=True, framealpha=0.9, edgecolor="none")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {output_path}")


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print(f"Layers: {LAYERS}")
    print(f"Models: {list(MODELS.keys())}\n")

    # Load ImageNet with original 1K labels
    use_split = VALIDATION_MODE
    _, loaders = get_obj_cls_loader(
        {"dataset": "imagenet", "batchsize": BATCH_SIZE,
         "num_workers": NUM_WORKERS, "pca_labels": False},
        shuffle=False, preprocess=True, train_test_split=use_split,
    )
    loader = loaders["test"] if use_split else loaders["all"]
    print(f"Dataset: {len(loader.dataset)} images "
          f"({'test split' if use_split else 'full'})\n")

    # Evaluate models
    all_results = []
    csi_per_model = {}  # model_name -> pooled CSI array
    n_layers = len(LAYERS)

    for model_name, model_cfg in MODELS.items():
        print(f"\n{'='*50}")
        print(f"Model: {model_name}")

        model = load_model(model_cfg["path"], device)

        # Single pass to accumulate 1K-level stats
        class_sums, class_counts = accumulate_activations(
            model, loader, LAYERS, device
        )

        # Compute CSI per layer and pool
        layer_csi_arrays = []
        for i, layer in enumerate(LAYERS):
            csi = compute_layer_csi(class_sums, class_counts, layer)
            layer_csi_arrays.append(csi)

            mean_csi = float(np.mean(csi))
            print(f"  {layer:6s}: mean_csi={mean_csi:.4f}  n_neurons={len(csi)}")

            all_results.append({
                "model": model_name,
                "layer": layer,
                "depth_normalized": i / (n_layers - 1),
                "mean_csi": mean_csi,
                "std_csi": float(np.std(csi)),
                "n_neurons": len(csi),
            })

        csi_per_model[model_name] = np.concatenate(layer_csi_arrays)

        del model
        torch.cuda.empty_cache()

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "class_selectivity_index.csv")
    plot_path = os.path.join(OUTPUT_DIR, "class_selectivity_index.png")

    df = pd.DataFrame(all_results)
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Summary table
    print("\n" + "="*60)
    print("MEAN CSI PER LAYER")
    print("="*60)
    pivot = df.pivot_table(
        values="mean_csi", index="layer", columns="model", aggfunc="mean",
    )
    # Reorder layers
    pivot = pivot.reindex(LAYERS)
    print(pivot.to_string(float_format="%.4f"))

    # Pooled stats
    print("\n" + "="*60)
    print("POOLED CSI (all layers)")
    print("="*60)
    for model_name, csi in csi_per_model.items():
        print(f"  {model_name}: mean={np.mean(csi):.4f}  "
              f"std={np.std(csi):.4f}  n={len(csi)}")

    # Plot
    plot_csi_histogram(csi_per_model, plot_path)


if __name__ == "__main__":
    main()
