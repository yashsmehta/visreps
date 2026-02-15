"""
Plot: Curriculum RSA comparison across 3 AlexNet variants.

Two-panel figure (EVC, VVS) comparing:
1. AlexNet trained from scratch on 1K-way
2. AlexNet trained on 64-way coarse labels
3. AlexNet 64-way pretrained, finetuned on 1K-way (curriculum, late_layers)

Run from repo root:
    python experiments/coarse_grain_benefits/plot_curriculum_rsa.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
INPUT_CSV = "experiments/coarse_grain_benefits/results/curriculum_nsd_rsa.csv"
OUTPUT_FILE = "experiments/coarse_grain_benefits/results/curriculum_rsa_comparison.png"

MODEL_NAMES = [
    "AlexNet (1K classes)",
    "AlexNet (64 classes)",
    "AlexNet (64→1K curriculum)",
]

# ─────────────────────────────────────────────────────────────
# NATURE-STYLE PLOT SETTINGS
# ─────────────────────────────────────────────────────────────
FIGURE_WIDTH_MM = 183
FIGURE_HEIGHT_MM = 70
MM_TO_INCH = 0.0393701

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 7,
    'axes.titlesize': 8,
    'axes.labelsize': 7,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'lines.linewidth': 1.5,
    'lines.markersize': 4,
    'axes.linewidth': 0.5,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 2,
    'ytick.major.size': 2,
    'axes.grid': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.02,
})

# Colorblind-friendly palette
COLORS = {
    "AlexNet (1K classes)": "#0072B2",           # Blue
    "AlexNet (64 classes)": "#2E8B57",           # Sea green
    "AlexNet (64→1K curriculum)": "#D55E00",     # Orange
}

MARKERS = {
    "AlexNet (1K classes)": "o",
    "AlexNet (64 classes)": "o",
    "AlexNet (64→1K curriculum)": "D",           # Diamond
}

LINESTYLES = {
    "AlexNet (1K classes)": "-",
    "AlexNet (64 classes)": "--",
    "AlexNet (64→1K curriculum)": "-.",
}


def plot_region(ax, df, region, ylabel=True):
    """Plot RSA vs depth for a single region."""
    lines = []
    labels = []

    for model_name in MODEL_NAMES:
        mask = (df["model_name"] == model_name) & (df["region"] == region)
        model_df = df[mask]

        if model_df.empty:
            print(f"Warning: No data for {model_name} in {region}")
            continue

        grouped = model_df.groupby("depth_normalized")["rsa_score"].mean()
        depths = grouped.index.values
        means = grouped.values

        sort_idx = np.argsort(depths)
        depths = depths[sort_idx]
        means = means[sort_idx]

        color = COLORS[model_name]
        marker = MARKERS[model_name]
        linestyle = LINESTYLES[model_name]

        line, = ax.plot(
            depths, means,
            color=color,
            marker=marker,
            linestyle=linestyle,
            markersize=4,
            markerfacecolor=color,
            markeredgecolor='white',
            markeredgewidth=0.4,
            linewidth=1.5,
            zorder=3,
        )
        lines.append(line)
        labels.append(model_name)

    ax.set_xlabel("Normalized depth")
    if ylabel:
        ax.set_ylabel("RSA score")

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, None)

    region_short = "Early Visual" if "early" in region.lower() else "Ventral Visual"
    ax.set_title(region_short, fontweight='bold', pad=4)

    return lines, labels


def main():
    df = pd.read_csv(INPUT_CSV)

    fig_width = FIGURE_WIDTH_MM * MM_TO_INCH
    fig_height = FIGURE_HEIGHT_MM * MM_TO_INCH
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))

    plt.subplots_adjust(wspace=0.2, left=0.07, right=0.72, top=0.88, bottom=0.18)

    lines1, labels1 = plot_region(axes[0], df, "early visual stream", ylabel=True)
    lines2, labels2 = plot_region(axes[1], df, "ventral visual stream", ylabel=False)

    # Deduplicated legend
    all_lines = []
    all_labels = []
    seen = set()
    for lines, labels in [(lines1, labels1), (lines2, labels2)]:
        for line, label in zip(lines, labels):
            if label not in seen:
                all_lines.append(line)
                all_labels.append(label)
                seen.add(label)

    fig.legend(
        all_lines, all_labels,
        loc='center right',
        bbox_to_anchor=(0.98, 0.5),
        frameon=True,
        framealpha=0.95,
        edgecolor='none',
        fancybox=False,
    )

    # Panel labels
    for i, ax in enumerate(axes):
        ax.text(
            -0.15, 1.08,
            chr(97 + i),
            transform=ax.transAxes,
            fontsize=10,
            fontweight='bold',
            va='top',
        )

    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format='png', dpi=300)
    print(f"Saved: {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
