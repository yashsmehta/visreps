"""
Corruption robustness on THINGS: tests whether the coarse (CLIP 4-class) model
maintains behavioral alignment under image corruption better than the 1K model.

Two metrics per condition:
  - Behavioral RSA: corr(corrupted_model_RDM, behavioral_RDM)
  - Representation stability: corr(corrupted_model_RDM, clean_model_RDM)

Outputs:
  - data/corruption_robustness.csv
  - figures/corruption_robustness.png

Run from project root:
  python experiments/things_visualizations/corruption_robustness.py
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

import numpy as np
np.float_ = np.float64  # imagecorruptions uses removed np.float_
import pandas as pd
import torch
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
from imagecorruptions import corrupt

from experiments.things_visualizations.extract_data import (
    load_model_from_checkpoint, make_extractor, concept_average, CHECKPOINTS, BEST_LAYERS,
)
from experiments.things_visualizations.utils import save_fig, FIG_DIR, COLOR_CLIP4, COLOR_1K
from visreps.analysis.rsa import compute_rdm, compute_rdm_correlation
from visreps.dataloaders.neural import load_things_data

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 256
DATA_DIR = os.path.join(PROJECT_ROOT, "experiments", "things_visualizations", "data")

MODELS = {
    "CLIP 4-class": {
        "checkpoint": CHECKPOINTS["clip-4"],
        "layer": BEST_LAYERS["clip-4"], "color": COLOR_CLIP4,
    },
    "1K-class": {
        "checkpoint": CHECKPOINTS["1000-class"],
        "layer": BEST_LAYERS["1000-class"], "color": COLOR_1K,
    },
}

CORRUPTIONS = ["gaussian_noise", "defocus_blur", "contrast", "fog", "pixelate"]
SEVERITIES = [1, 2, 3, 4, 5]

PRE_TRANSFORM = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
NORMALIZE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ─── Data ─────────────────────────────────────────────────────────────


def load_things_images():
    """Load THINGS images as pre-processed 224x224 numpy arrays."""
    targets, img_paths = load_things_data()
    stimulus_ids = sorted(img_paths.keys())
    print(f"Loading {len(stimulus_ids)} THINGS images...")
    image_arrays = []
    for sid in tqdm(stimulus_ids, desc="Pre-processing", leave=False):
        img = Image.open(img_paths[sid]).convert("RGB")
        image_arrays.append(np.array(PRE_TRANSFORM(img)))
    return image_arrays, stimulus_ids, targets


# ─── Feature extraction ──────────────────────────────────────────────


def extract_features(extractor, image_arrays, layer, corruption=None, severity=3):
    """Extract features from (optionally corrupted) numpy arrays in batches."""
    all_features = []
    with torch.no_grad():
        for i in range(0, len(image_arrays), BATCH_SIZE):
            batch_arrays = image_arrays[i : i + BATCH_SIZE]
            tensors = []
            for arr in batch_arrays:
                if corruption:
                    arr = corrupt(arr, corruption_name=corruption, severity=severity)
                tensors.append(NORMALIZE(Image.fromarray(arr.astype(np.uint8))))
            feats = extractor(torch.stack(tensors).to(DEVICE))[layer]
            all_features.append(feats.view(feats.size(0), -1).cpu().numpy())
    return np.vstack(all_features)


# ─── Plotting ────────────────────────────────────────────────────────


def plot_results(df):
    """2x5 grid: behavioral RSA (top) and representation stability (bottom)."""
    import seaborn as sns
    from matplotlib.lines import Line2D

    sns.set_theme(style="ticks", context="paper", font_scale=1.0)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 3, "ytick.major.size": 3,
    })

    # Refined palette — colorblind-friendly, publication-quality
    C_CLIP4 = "#0072B2"   # blue
    C_1K = "#D55E00"      # vermillion
    model_colors = {"CLIP 4-class": C_CLIP4, "1K-class": C_1K}

    n_corr = len(CORRUPTIONS)
    fig, axes = plt.subplots(
        2, n_corr, figsize=(2.4 * n_corr, 4.4),
        sharex=True, sharey="row",
    )

    metrics = [
        ("behavioral_rsa", "Behavioral RSA ($\\rho_s$)"),
        ("stability", "Representation stability ($\\rho_s$)"),
    ]
    row_labels = ["a", "b"]

    baselines = {name: df[(df["model"] == name) & (df["corruption"] == "clean")]
                 for name in model_colors}

    for row, (metric, ylabel) in enumerate(metrics):
        for col, corr_name in enumerate(CORRUPTIONS):
            ax = axes[row, col]

            # Light grid
            ax.grid(axis="y", color="#ebebeb", lw=0.4, zorder=0)

            for name, color in model_colors.items():
                sub = (df[(df["model"] == name) & (df["corruption"] == corr_name)]
                       .sort_values("severity"))
                ax.plot(sub["severity"], sub[metric],
                        "-o", color=color, lw=1.5, ms=5,
                        markeredgecolor="white", markeredgewidth=0.7,
                        zorder=3, clip_on=False)
                # Baseline (clean performance)
                bval = baselines[name][metric].values[0]
                ax.axhline(bval, color=color, ls=(0, (4, 3)), lw=0.7, alpha=0.55, zorder=1)

            # Column titles (top row only)
            if row == 0:
                title = corr_name.replace("_", " ").title()
                ax.set_title(title, fontsize=8, fontweight="semibold", pad=5)

            # X-axis label (bottom center panel only)
            if row == 1 and col == n_corr // 2:
                ax.set_xlabel("Corruption severity", fontsize=8, labelpad=5)

            ax.set_xticks(SEVERITIES)
            ax.tick_params(labelsize=6.5, pad=2)

            # Y-axis label (left column only)
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=7.5, labelpad=3)

            # Despine with offset
            sns.despine(ax=ax, offset=3)

        # Row label (panel letter)
        axes[row, 0].text(-0.32, 1.10, row_labels[row],
                          transform=axes[row, 0].transAxes,
                          fontsize=11, fontweight="bold", va="top")

    # Legend — upper right of first panel, compact
    legend_elements = [
        Line2D([0], [0], color=C_CLIP4, marker="o", lw=1.5, ms=5,
               markeredgecolor="white", markeredgewidth=0.7, label="CLIP 4-class"),
        Line2D([0], [0], color=C_1K, marker="o", lw=1.5, ms=5,
               markeredgecolor="white", markeredgewidth=0.7, label="1000-class"),
    ]
    # Legend with baseline explanation
    from matplotlib.lines import Line2D as L2D
    baseline_handle = L2D([0], [0], color="#999999", ls=(0, (4, 3)), lw=0.8,
                           label="Clean baseline")
    all_handles = legend_elements + [baseline_handle]
    axes[0, -1].legend(handles=all_handles, fontsize=6.5, frameon=True,
                        loc="upper right", handletextpad=0.4, borderpad=0.45,
                        labelspacing=0.35, fancybox=False,
                        edgecolor="#dddddd", facecolor="white", framealpha=0.92)

    plt.tight_layout(h_pad=1.2, w_pad=0.5)
    fig.savefig(os.path.join(FIG_DIR, "corruption_robustness.png"),
                dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved: {os.path.join(FIG_DIR, 'corruption_robustness.png')}")


# ─── Main ────────────────────────────────────────────────────────────


def main():
    image_arrays, stimulus_ids, targets = load_things_images()
    concept_image_ids = targets["image_ids"]
    concept_names = sorted(targets["embeddings"].keys())

    emb_matrix = np.array([targets["embeddings"][c] for c in concept_names])
    behavioral_rdm = compute_rdm(torch.tensor(emb_matrix, dtype=torch.float32))
    print(f"Behavioral RDM: {behavioral_rdm.shape}")

    results = []

    for model_name, mcfg in MODELS.items():
        print(f"\n{'=' * 60}\n{model_name}\n{'=' * 60}")

        layer = mcfg["layer"]
        base_layer = layer.replace("_pre", "").replace("_post", "")
        model = load_model_from_checkpoint(mcfg["checkpoint"])
        extractor = make_extractor(model, [base_layer])

        # Clean baseline
        clean_feats = extract_features(extractor, image_arrays, layer)
        clean_acts, cnames = concept_average(clean_feats, stimulus_ids, concept_image_ids)
        assert cnames == concept_names

        clean_rdm = compute_rdm(torch.tensor(clean_acts, dtype=torch.float32))
        clean_rsa = compute_rdm_correlation(clean_rdm, behavioral_rdm, correlation="Spearman")
        print(f"  Clean RSA: {clean_rsa:.4f}")

        results.append({
            "model": model_name, "corruption": "clean", "severity": 0,
            "behavioral_rsa": clean_rsa, "stability": 1.0,
        })

        # Corrupted conditions
        for corr_name in CORRUPTIONS:
            for sev in SEVERITIES:
                feats = extract_features(extractor, image_arrays, layer, corr_name, sev)
                acts, _ = concept_average(feats, stimulus_ids, concept_image_ids)
                rdm = compute_rdm(torch.tensor(acts, dtype=torch.float32))

                rsa = compute_rdm_correlation(rdm, behavioral_rdm, correlation="Spearman")
                stab = compute_rdm_correlation(rdm, clean_rdm, correlation="Spearman")
                print(f"  {corr_name} s{sev}: RSA={rsa:.4f}  stab={stab:.4f}")

                results.append({
                    "model": model_name, "corruption": corr_name, "severity": sev,
                    "behavioral_rsa": rsa, "stability": stab,
                })

        del model, extractor
        torch.cuda.empty_cache()

    df = pd.DataFrame(results)
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, "corruption_robustness.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    plot_results(df)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot-only", action="store_true",
                        help="Re-plot from existing CSV without recomputing")
    args = parser.parse_args()

    if args.plot_only:
        os.makedirs(FIG_DIR, exist_ok=True)
        csv_path = os.path.join(DATA_DIR, "corruption_robustness.csv")
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from {csv_path}")
        plot_results(df)
    else:
        main()
