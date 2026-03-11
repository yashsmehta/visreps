"""Alternative per-layer profile visualizations for NSD and TVSD datasets.

Generates 5 plot types x 2 datasets = 10 figures total.
Must be run from the project root.
"""

import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sys.path.insert(0, "manuscript/figures")
from fig_utils import (
    GRAN_CFGS, GRAN_COLORS, GRAN_MARKERS,
    LAYER_ORDER_FULL, LAYER_LABEL_POSITIONS, LAYER_LABELS_SHORT,
    setup_style, find_best_architecture, fetch_layer_scores,
)

OUT_DIR = "manuscript/figures/misc"

DATASET_CONFIGS = {
    "nsd": {
        "regions": ["early visual stream", "ventral visual stream"],
        "region_labels": ["Early Visual Stream", "Ventral Visual Stream"],
    },
    "tvsd": {
        "regions": ["V1", "IT"],
        "region_labels": ["V1", "IT"],
    },
}

# Post-ReLU layer indices and their x-positions (0-indexed within post-only)
POST_RELU_LAYERS = [l for l in LAYER_ORDER_FULL if l.endswith("_post")]


def _get_data(neural_dataset, region):
    """Fetch layer scores for a given dataset and region."""
    pca_folder, arch_display = find_best_architecture(neural_dataset, region)
    all_scores = fetch_layer_scores(neural_dataset, region, pca_folder)
    return all_scores, pca_folder, arch_display


def _get_full_layer_arrays(all_scores, cfg_id):
    """Return x-positions and y-values for all available layers (full 14)."""
    scores = all_scores.get(cfg_id, {})
    layers = [l for l in LAYER_ORDER_FULL if l in scores]
    if not layers:
        return np.array([]), np.array([])
    x = np.array([LAYER_ORDER_FULL.index(l) for l in layers])
    y = np.array([scores[l] for l in layers])
    return x, y


def _get_post_relu_arrays(all_scores, cfg_id):
    """Return indices (0..6) and y-values for post-ReLU layers only."""
    scores = all_scores.get(cfg_id, {})
    x_vals, y_vals = [], []
    for i, layer in enumerate(POST_RELU_LAYERS):
        if layer in scores:
            x_vals.append(i)
            y_vals.append(scores[layer])
    return np.array(x_vals), np.array(y_vals)


def _format_x_full(ax, show_xlabel=True):
    """Format x-axis for all 14 layers."""
    ax.set_xticks(range(len(LAYER_ORDER_FULL)))
    tick_labels = [""] * len(LAYER_ORDER_FULL)
    for pos, label in zip(LAYER_LABEL_POSITIONS, LAYER_LABELS_SHORT):
        tick_labels[pos] = label
    ax.set_xticklabels(tick_labels, fontsize=7.5)
    ax.set_xlim(-0.5, len(LAYER_ORDER_FULL) - 0.5)
    if show_xlabel:
        ax.set_xlabel("Layer", fontsize=9, labelpad=4)


def _format_x_post(ax, show_xlabel=True):
    """Format x-axis for 7 post-ReLU layers."""
    ax.set_xticks(range(7))
    ax.set_xticklabels(LAYER_LABELS_SHORT, fontsize=7.5)
    ax.set_xlim(-0.5, 6.5)
    if show_xlabel:
        ax.set_xlabel("Layer", fontsize=9, labelpad=4)


def _gran_label(cfg_id):
    return f"{cfg_id}-way"


# ── Plot 1: Heatmap ──────────────────────────────────────────────────────

def plot_heatmap(neural_dataset):
    """Heatmap: granularity (y) x layer (x), color = score."""
    cfg = DATASET_CONFIGS[neural_dataset]
    n_regions = len(cfg["regions"])

    fig, axes = plt.subplots(n_regions, 1, figsize=(7, 2.2 * n_regions),
                             squeeze=False)

    for row, (region, region_label) in enumerate(zip(cfg["regions"], cfg["region_labels"])):
        ax = axes[row, 0]
        all_scores, pca_folder, arch_display = _get_data(neural_dataset, region)

        # Build matrix: rows = granularity, cols = post-ReLU layers
        matrix = np.full((len(GRAN_CFGS), 7), np.nan)
        for r, cfg_id in enumerate(GRAN_CFGS):
            _, y = _get_post_relu_arrays(all_scores, cfg_id)
            x_idx, _ = _get_post_relu_arrays(all_scores, cfg_id)
            for xi, yi in zip(x_idx.astype(int), y):
                matrix[r, xi] = yi

        im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", interpolation="nearest")

        # Annotate cells
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                val = matrix[r, c]
                if not np.isnan(val):
                    text_color = "white" if val > (np.nanmax(matrix) + np.nanmin(matrix)) / 2 else "black"
                    ax.text(c, r, f"{val:.3f}", ha="center", va="center",
                            fontsize=6, color=text_color)

        ax.set_xticks(range(7))
        ax.set_xticklabels(LAYER_LABELS_SHORT, fontsize=7.5)
        ax.set_yticks(range(len(GRAN_CFGS)))
        ax.set_yticklabels([str(c) for c in GRAN_CFGS], fontsize=7.5)
        ax.set_ylabel("Granularity", fontsize=9, labelpad=4)
        if row == n_regions - 1:
            ax.set_xlabel("Layer", fontsize=9, labelpad=4)
        ax.set_title(f"{region_label} ({arch_display})", fontsize=10,
                     fontweight="semibold", pad=6)

        cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        cbar.set_label(r"Spearman $\rho$", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    fig.tight_layout(h_pad=1.5)
    out_path = os.path.join(OUT_DIR, f"alt_perlayer_heatmap_{neural_dataset}.png")
    fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Plot 2: Area chart ───────────────────────────────────────────────────

def plot_area(neural_dataset):
    """Overlapping filled area chart for each granularity level."""
    cfg = DATASET_CONFIGS[neural_dataset]
    n_regions = len(cfg["regions"])

    fig, axes = plt.subplots(1, n_regions, figsize=(5.5 * n_regions, 3.5),
                             squeeze=False)

    for col, (region, region_label) in enumerate(zip(cfg["regions"], cfg["region_labels"])):
        ax = axes[0, col]
        all_scores, pca_folder, arch_display = _get_data(neural_dataset, region)

        # Plot from lowest to highest granularity so higher ones overlay
        for cfg_id in GRAN_CFGS:
            x, y = _get_full_layer_arrays(all_scores, cfg_id)
            if len(x) == 0:
                continue
            color = GRAN_COLORS[cfg_id]
            ax.fill_between(x, 0, y, alpha=0.25, color=color, zorder=1)
            ax.plot(x, y, color=color, linewidth=1.5, label=_gran_label(cfg_id),
                    zorder=2, alpha=0.9)

        # Untrained
        x_un, y_un = _get_full_layer_arrays(all_scores, 0)
        if len(x_un) > 0:
            ax.plot(x_un, y_un, color="#AAAAAA", linewidth=1.1, linestyle="--",
                    label="Untrained", zorder=3)

        _format_x_full(ax)
        ax.set_ylabel(r"Spearman $\rho$", fontsize=9, labelpad=4)
        ax.set_title(f"{region_label}", fontsize=10, fontweight="semibold", pad=6)
        ax.yaxis.grid(True, color="#EBEBEB", linewidth=0.4, zorder=0)
        sns.despine(ax=ax, right=True, top=True, offset=4)

    # Legend from first axis
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   ncol=min(len(handles), 8), fontsize=7.5,
                   frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out_path = os.path.join(OUT_DIR, f"alt_perlayer_area_{neural_dataset}.png")
    fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Plot 3: Radar / spider chart ─────────────────────────────────────────

def plot_radar(neural_dataset):
    """Radar/spider chart with post-ReLU layers as axes."""
    cfg = DATASET_CONFIGS[neural_dataset]
    n_regions = len(cfg["regions"])
    n_spokes = 7
    angles = np.linspace(0, 2 * np.pi, n_spokes, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, axes = plt.subplots(1, n_regions, figsize=(5 * n_regions, 5),
                             subplot_kw=dict(polar=True), squeeze=False)

    for col, (region, region_label) in enumerate(zip(cfg["regions"], cfg["region_labels"])):
        ax = axes[0, col]
        all_scores, pca_folder, arch_display = _get_data(neural_dataset, region)

        for cfg_id in GRAN_CFGS:
            _, y = _get_post_relu_arrays(all_scores, cfg_id)
            if len(y) != 7:
                continue
            values = y.tolist() + [y[0]]  # close
            color = GRAN_COLORS[cfg_id]
            ax.plot(angles, values, color=color, linewidth=1.4, label=_gran_label(cfg_id))
            ax.fill(angles, values, color=color, alpha=0.08)

        # Untrained
        scores_un = all_scores.get(0, {})
        y_un = []
        for layer in POST_RELU_LAYERS:
            y_un.append(scores_un.get(layer, np.nan))
        y_un = np.array(y_un)
        if not np.all(np.isnan(y_un)):
            vals = y_un.tolist() + [y_un[0]]
            ax.plot(angles, vals, color="#AAAAAA", linewidth=1.1, linestyle="--",
                    label="Untrained")

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(LAYER_LABELS_SHORT, fontsize=8)
        ax.set_title(f"{region_label}", fontsize=10, fontweight="semibold", pad=15)

        # Clean up radial grid
        ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.4)
        ax.xaxis.grid(True, color="#DDDDDD", linewidth=0.4)
        ax.set_facecolor("white")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center",
                   ncol=min(len(handles), 8), fontsize=7.5,
                   frameon=False, bbox_to_anchor=(0.5, -0.02))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    out_path = os.path.join(OUT_DIR, f"alt_perlayer_radar_{neural_dataset}.png")
    fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Plot 4: Difference from 1000-way baseline ────────────────────────────

def plot_diff(neural_dataset):
    """Difference plot: coarse - 1000-way. Green = outperforms, red = underperforms."""
    cfg = DATASET_CONFIGS[neural_dataset]
    n_regions = len(cfg["regions"])
    coarse_cfgs = [c for c in GRAN_CFGS if c != 1000]

    fig, axes = plt.subplots(len(coarse_cfgs), n_regions,
                             figsize=(5.5 * n_regions, 1.8 * len(coarse_cfgs)),
                             sharex=True, squeeze=False)

    for col, (region, region_label) in enumerate(zip(cfg["regions"], cfg["region_labels"])):
        all_scores, pca_folder, arch_display = _get_data(neural_dataset, region)

        # Get 1000-way baseline
        x_base, y_base = _get_full_layer_arrays(all_scores, 1000)
        if len(x_base) == 0:
            continue
        base_dict = dict(zip(x_base.astype(int), y_base))

        for row, cfg_id in enumerate(coarse_cfgs):
            ax = axes[row, col]
            x, y = _get_full_layer_arrays(all_scores, cfg_id)
            if len(x) == 0:
                continue

            # Compute difference only at shared positions
            diff_x, diff_y = [], []
            for xi, yi in zip(x.astype(int), y):
                if xi in base_dict:
                    diff_x.append(xi)
                    diff_y.append(yi - base_dict[xi])
            diff_x = np.array(diff_x)
            diff_y = np.array(diff_y)

            ax.axhline(0, color="#888888", linewidth=0.8, linestyle="-", zorder=1)
            ax.plot(diff_x, diff_y, color=GRAN_COLORS[cfg_id], linewidth=1.3, zorder=3)

            # Fill positive green, negative red
            ax.fill_between(diff_x, 0, diff_y,
                            where=diff_y >= 0, color="#2ca02c", alpha=0.3,
                            interpolate=True, zorder=2)
            ax.fill_between(diff_x, 0, diff_y,
                            where=diff_y < 0, color="#d62728", alpha=0.3,
                            interpolate=True, zorder=2)

            if col == 0:
                ax.set_ylabel(f"{cfg_id}-way", fontsize=8, labelpad=4)
            if row == 0:
                ax.set_title(f"{region_label}", fontsize=10,
                             fontweight="semibold", pad=6)

            ax.yaxis.grid(True, color="#EBEBEB", linewidth=0.4, zorder=0)
            sns.despine(ax=ax, right=True, top=True, offset=2)

            if row == len(coarse_cfgs) - 1:
                _format_x_full(ax, show_xlabel=True)
            else:
                _format_x_full(ax, show_xlabel=False)
                ax.set_xticklabels([])

    # Shared y-label
    fig.supylabel(r"$\Delta$ Spearman $\rho$ (coarse $-$ 1000-way)", fontsize=9, x=0.01)
    fig.tight_layout(rect=[0.03, 0, 1, 1], h_pad=0.5)
    out_path = os.path.join(OUT_DIR, f"alt_perlayer_diff_{neural_dataset}.png")
    fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Plot 5: Ridge / joy plot ─────────────────────────────────────────────

def plot_ridge(neural_dataset):
    """Ridge (joy) plot: vertically offset line plots for each granularity."""
    cfg = DATASET_CONFIGS[neural_dataset]
    n_regions = len(cfg["regions"])

    # All granularity levels + untrained, bottom to top
    plot_cfgs = GRAN_CFGS  # 2, 4, 8, 16, 32, 64, 1000

    fig, axes = plt.subplots(1, n_regions, figsize=(5.5 * n_regions, 5),
                             squeeze=False)

    for col, (region, region_label) in enumerate(zip(cfg["regions"], cfg["region_labels"])):
        ax = axes[0, col]
        all_scores, pca_folder, arch_display = _get_data(neural_dataset, region)

        # Determine vertical offset spacing
        all_vals = []
        for cfg_id in plot_cfgs:
            _, y = _get_post_relu_arrays(all_scores, cfg_id)
            if len(y) > 0:
                all_vals.extend(y.tolist())
        if not all_vals:
            continue
        val_range = max(all_vals) - min(all_vals)
        offset_step = val_range * 0.6

        for i, cfg_id in enumerate(plot_cfgs):
            x, y = _get_post_relu_arrays(all_scores, cfg_id)
            if len(x) == 0:
                continue
            offset = i * offset_step
            color = GRAN_COLORS[cfg_id]
            y_offset = y + offset

            # Filled area
            ax.fill_between(x, offset, y_offset, alpha=0.35, color=color, zorder=i + 1)
            ax.plot(x, y_offset, color=color, linewidth=1.4, zorder=i + 2)

            # Label on the left
            ax.text(-0.7, offset + offset_step * 0.15, _gran_label(cfg_id),
                    fontsize=7, color=color, fontweight="semibold",
                    ha="right", va="bottom")

        _format_x_post(ax)
        ax.set_title(f"{region_label}", fontsize=10, fontweight="semibold", pad=6)
        ax.set_yticks([])
        ax.set_ylabel("")
        sns.despine(ax=ax, left=True, right=True, top=True, offset=4)

    fig.tight_layout()
    out_path = os.path.join(OUT_DIR, f"alt_perlayer_ridge_{neural_dataset}.png")
    fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    setup_style()

    for dataset in ["nsd", "tvsd"]:
        print(f"\n{'='*60}")
        print(f"  Generating alternative per-layer plots for {dataset.upper()}")
        print(f"{'='*60}")

        plot_heatmap(dataset)
        plot_area(dataset)
        plot_radar(dataset)
        plot_diff(dataset)
        plot_ridge(dataset)

    print("\nDone. All plots saved to:", OUT_DIR)


if __name__ == "__main__":
    main()
