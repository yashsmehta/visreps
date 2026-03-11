"""Alternative visualizations of the reconstruction analysis data.

Generates 3 plot types x 2 datasets (NSD, TVSD) = 6 figures:
  1. Bar chart at key PCs (alt_recon_bars_{dataset}.png)
  2. Difference curve (alt_recon_diff_{dataset}.png)
  3. Scatter plot (alt_recon_scatter_{dataset}.png)

Run from project root:
    python manuscript/figures/misc/alt_reconstruction.py
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

sys.path.insert(0, "experiments/reconstruction_analysis")
from plot_utils import (
    query_reconstruction_curve,
    query_untrained_baseline,
    aggregate_curve,
)

sys.path.insert(0, "manuscript/figures")
from fig_utils import GRAN_COLORS, setup_style

# ── Output directory ─────────────────────────────────────────────────────────
OUT_DIR = Path("manuscript/figures/misc")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Colors ───────────────────────────────────────────────────────────────────
FINE_COLOR = "#e6a200"       # golden amber for 1000-way
UNTRAINED_COLOR = "#969696"  # neutral grey

# ── Region configs: which coarse model to compare per region ─────────────────
NSD_COARSE_CONFIG = {
    "early visual stream": (64, "/data/ymehta3/alexnet_pca"),
    "ventral visual stream": (16, "/data/ymehta3/clip_pca"),
}
TVSD_COARSE_CONFIG = {
    "V1": (64, "/data/ymehta3/alexnet_pca"),
    "IT": (64, "/data/ymehta3/alexnet_pca"),
}

DATASET_CONFIGS = {
    "nsd": {
        "regions": ["early visual stream", "ventral visual stream"],
        "labels": {"early visual stream": "Early Visual Stream",
                   "ventral visual stream": "Ventral Visual Stream"},
        "coarse_config": NSD_COARSE_CONFIG,
    },
    "tvsd": {
        "regions": ["V1", "IT"],
        "labels": {"V1": "V1", "IT": "IT"},
        "coarse_config": TVSD_COARSE_CONFIG,
    },
}

# Key PC counts for bar chart
KEY_PCS = [1, 5, 10, 50]


# ── Data fetching ────────────────────────────────────────────────────────────

def fetch_data(neural_dataset, region, coarse_config):
    """Fetch fine, coarse, and untrained data for a single region."""
    fine_df = query_reconstruction_curve(neural_dataset, region)
    fine_agg = aggregate_curve(fine_df)

    cfg_id, checkpoint_dir = coarse_config[region]
    coarse_df = query_reconstruction_curve(
        neural_dataset, region, cfg_id=cfg_id, checkpoint_dir=checkpoint_dir,
    )
    coarse_agg = aggregate_curve(coarse_df)

    untrained = query_untrained_baseline(neural_dataset, region)

    return fine_agg, coarse_agg, untrained, cfg_id


# ── Plot 1: Bar chart at key PCs ────────────────────────────────────────────

def _add_rounded_bar(ax, x, y, width, color, label=None):
    """Add a single bar with rounded top corners using FancyBboxPatch."""
    if np.isnan(y) or y <= 0:
        return
    bar = FancyBboxPatch(
        (x - width / 2, 0), width, y,
        boxstyle="round,pad=0,rounding_size=0.008",
        facecolor=color, edgecolor="white", linewidth=0.6,
        zorder=3, label=label,
    )
    ax.add_patch(bar)


def plot_bars(neural_dataset):
    """Bar chart at key PC counts comparing 1000-way and coarse reconstruction."""
    dc = DATASET_CONFIGS[neural_dataset]
    regions = dc["regions"]
    n_regions = len(regions)

    setup_style()
    fig, axes = plt.subplots(1, n_regions, figsize=(4.0 * n_regions, 3.5), squeeze=False)
    axes = axes.flatten()

    bar_width = 0.35
    group_spacing = 1.2

    for ax_idx, region in enumerate(regions):
        ax = axes[ax_idx]
        fine_agg, coarse_agg, untrained, cfg_id = fetch_data(
            neural_dataset, region, dc["coarse_config"],
        )

        coarse_color = GRAN_COLORS.get(cfg_id, "#2166ac")
        x_positions = np.arange(len(KEY_PCS)) * group_spacing

        for i, k in enumerate(KEY_PCS):
            # Fine (1000-way) bar
            fine_row = fine_agg[fine_agg["pca_k"] == k]
            fine_val = fine_row["mean"].values[0] if not fine_row.empty else np.nan
            fine_lo = fine_row["ci_low"].values[0] if not fine_row.empty else np.nan
            fine_hi = fine_row["ci_high"].values[0] if not fine_row.empty else np.nan

            label_fine = "1000-way" if i == 0 else None
            _add_rounded_bar(ax, x_positions[i] - bar_width / 2, fine_val,
                             bar_width, FINE_COLOR, label=label_fine)
            if not np.isnan(fine_lo):
                ax.errorbar(x_positions[i] - bar_width / 2, fine_val,
                            yerr=[[fine_val - fine_lo], [fine_hi - fine_val]],
                            fmt="none", ecolor="#333333", elinewidth=0.8,
                            capsize=2, capthick=0.8, zorder=4)

            # Coarse bar
            coarse_row = coarse_agg[coarse_agg["pca_k"] == k]
            coarse_val = coarse_row["mean"].values[0] if not coarse_row.empty else np.nan
            coarse_lo = coarse_row["ci_low"].values[0] if not coarse_row.empty else np.nan
            coarse_hi = coarse_row["ci_high"].values[0] if not coarse_row.empty else np.nan

            label_coarse = f"{cfg_id}-way" if i == 0 else None
            _add_rounded_bar(ax, x_positions[i] + bar_width / 2, coarse_val,
                             bar_width, coarse_color, label=label_coarse)
            if not np.isnan(coarse_lo):
                ax.errorbar(x_positions[i] + bar_width / 2, coarse_val,
                            yerr=[[coarse_val - coarse_lo], [coarse_hi - coarse_val]],
                            fmt="none", ecolor="#333333", elinewidth=0.8,
                            capsize=2, capthick=0.8, zorder=4)

        # Untrained baseline
        un_mean, un_lo, un_hi = untrained
        if not np.isnan(un_mean):
            ax.axhline(un_mean, color=UNTRAINED_COLOR, linestyle="--",
                       linewidth=1.3, label="Untrained", zorder=2)

        ax.set_xticks(x_positions)
        ax.set_xticklabels([f"k={k}" for k in KEY_PCS], fontsize=8)
        ax.set_xlabel("Number of PCs", fontsize=9, labelpad=4)
        if ax_idx == 0:
            ax.set_ylabel(r"Spearman $\rho$", fontsize=9, labelpad=4)
        ax.set_title(dc["labels"][region], fontsize=10, fontweight="semibold", pad=6)
        ax.set_xlim(x_positions[0] - 0.8, x_positions[-1] + 0.8)
        ax.autoscale_view(scalex=False)
        sns.despine(ax=ax, right=True, top=True, offset=4)
        ax.legend(fontsize=7.5, frameon=True, edgecolor="#cccccc", fancybox=False,
                  loc="upper left")

    plt.tight_layout()
    out_path = OUT_DIR / f"alt_recon_bars_{neural_dataset}.png"
    plt.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Plot 2: Difference curve ────────────────────────────────────────────────

def plot_diff(neural_dataset):
    """Difference curve: coarse - fine reconstruction score as f(k)."""
    dc = DATASET_CONFIGS[neural_dataset]
    regions = dc["regions"]
    n_regions = len(regions)

    setup_style()
    fig, axes = plt.subplots(1, n_regions, figsize=(4.0 * n_regions, 3.2), squeeze=False)
    axes = axes.flatten()

    for ax_idx, region in enumerate(regions):
        ax = axes[ax_idx]
        fine_agg, coarse_agg, _, cfg_id = fetch_data(
            neural_dataset, region, dc["coarse_config"],
        )

        if fine_agg.empty or coarse_agg.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="#888")
            ax.set_title(dc["labels"][region], fontsize=10, fontweight="semibold")
            continue

        # Merge on pca_k to get aligned values
        merged = fine_agg.merge(coarse_agg, on="pca_k", suffixes=("_fine", "_coarse"))
        k = merged["pca_k"].values
        diff = merged["mean_coarse"].values - merged["mean_fine"].values

        # Zero line
        ax.axhline(0, color="#333333", linewidth=0.8, linestyle="-", zorder=1)

        # Fill positive (coarse > fine) in blue, negative in red
        ax.fill_between(k, 0, diff, where=(diff >= 0),
                        color="#2166ac", alpha=0.3, interpolate=True, zorder=2,
                        label=f"{cfg_id}-way > 1000-way recon.")
        ax.fill_between(k, 0, diff, where=(diff < 0),
                        color="#d6604d", alpha=0.3, interpolate=True, zorder=2,
                        label=f"1000-way recon. > {cfg_id}-way")

        # Line
        ax.plot(k, diff, "-o", color="#333333", markersize=4, linewidth=1.4,
                markeredgecolor="white", markeredgewidth=0.5, zorder=3)

        ax.set_xlabel("Number of PCs ($k$)", fontsize=9, labelpad=4)
        if ax_idx == 0:
            ax.set_ylabel(r"$\Delta$ Spearman $\rho$", fontsize=9, labelpad=4)
        ax.set_title(dc["labels"][region], fontsize=10, fontweight="semibold", pad=6)
        ax.set_xticks(k)
        labeled = {1, 5, 10, 20, 30, 40, 50} | {int(k[0]), int(k[-1])}
        ax.set_xticklabels(
            [str(int(v)) if int(v) in labeled else "" for v in k], fontsize=8,
        )
        ax.tick_params(axis="both", which="major", labelsize=8, length=4,
                       width=0.8, direction="out")
        sns.despine(ax=ax, right=True, top=True, offset=4)
        ax.legend(fontsize=7, frameon=True, edgecolor="#cccccc", fancybox=False,
                  loc="best")

    plt.tight_layout()
    out_path = OUT_DIR / f"alt_recon_diff_{neural_dataset}.png"
    plt.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Plot 3: Scatter plot ────────────────────────────────────────────────────

def plot_scatter(neural_dataset):
    """Scatter: x = reconstructed 1000-way score, y = coarse score, per k."""
    dc = DATASET_CONFIGS[neural_dataset]
    regions = dc["regions"]
    n_regions = len(regions)

    setup_style()
    fig, axes = plt.subplots(1, n_regions, figsize=(4.2 * n_regions, 4.0), squeeze=False)
    axes = axes.flatten()

    for ax_idx, region in enumerate(regions):
        ax = axes[ax_idx]
        fine_agg, coarse_agg, _, cfg_id = fetch_data(
            neural_dataset, region, dc["coarse_config"],
        )

        if fine_agg.empty or coarse_agg.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12, color="#888")
            ax.set_title(dc["labels"][region], fontsize=10, fontweight="semibold")
            continue

        merged = fine_agg.merge(coarse_agg, on="pca_k", suffixes=("_fine", "_coarse"))
        x = merged["mean_fine"].values
        y = merged["mean_coarse"].values
        k_vals = merged["pca_k"].values.astype(int)

        # Color and size by k
        max_k = max(k_vals) if len(k_vals) > 0 else 1
        sizes = 30 + 120 * (k_vals / max_k)  # scale size by k
        colors = plt.cm.viridis(k_vals / max_k)

        # Diagonal reference line
        all_vals = np.concatenate([x, y])
        lim_lo = min(all_vals) - 0.01
        lim_hi = max(all_vals) + 0.01
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "--", color="#999999",
                linewidth=1.0, zorder=1, label="y = x")

        # Scatter points
        sc = ax.scatter(x, y, s=sizes, c=k_vals, cmap="viridis", edgecolors="white",
                        linewidths=0.6, zorder=3)

        # Label each point with its k value
        for xi, yi, ki in zip(x, y, k_vals):
            ax.annotate(str(ki), (xi, yi), textcoords="offset points",
                        xytext=(5, 5), fontsize=7, color="#333333", zorder=4)

        ax.set_xlabel("1000-way reconstructed score", fontsize=9, labelpad=4)
        if ax_idx == 0:
            ax.set_ylabel(f"{cfg_id}-way score", fontsize=9, labelpad=4)
        else:
            ax.set_ylabel(f"{cfg_id}-way score", fontsize=9, labelpad=4)
        ax.set_title(dc["labels"][region], fontsize=10, fontweight="semibold", pad=6)
        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(axis="both", which="major", labelsize=8, length=4,
                       width=0.8, direction="out")
        sns.despine(ax=ax, right=True, top=True, offset=4)

        # Colorbar
        cbar = plt.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label("$k$ (PCs)", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        ax.legend(fontsize=7.5, frameon=True, edgecolor="#cccccc", fancybox=False,
                  loc="upper left")

    plt.tight_layout()
    out_path = OUT_DIR / f"alt_recon_scatter_{neural_dataset}.png"
    plt.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    for dataset in ["nsd", "tvsd"]:
        print(f"\n{'='*60}")
        print(f"  Dataset: {dataset.upper()}")
        print(f"{'='*60}")
        plot_bars(dataset)
        plot_diff(dataset)
        plot_scatter(dataset)

    print("\nDone. All figures saved to manuscript/figures/misc/")


if __name__ == "__main__":
    main()
