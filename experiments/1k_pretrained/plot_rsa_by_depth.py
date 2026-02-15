"""
Publication-quality plot: RSA scores by normalized depth across architectures.

Creates a two-panel figure comparing early and ventral visual streams.
Styled for Nature journal publication standards.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
INPUT_CSV = "logs/1k_pretrained_nsd_rsa.csv"
OUTPUT_FILE = "experiments/1k_pretrained/rsa_depth_comparison.png"

# Models to plot per region: (model, model_source, display_name)
MODELS_EVS = [
    ("alexnet", "/data/ymehta3/default", "AlexNet (1K classes)"),
    ("vit", "torchvision", "ViT-B/16 (1K classes)"),
    ("alexnet", "/data/ymehta3/alexnet_pca", "AlexNet Coarse\n(64 classes, AlexNet reps)"),
]

MODELS_VVS = [
    ("alexnet", "/data/ymehta3/default", "AlexNet (1K classes)"),
    ("vit", "torchvision", "ViT-B/16 (1K classes)"),
    ("alexnet", "/data/ymehta3/vit_pca", "AlexNet Coarse\n(64 classes, ViT reps)"),
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

# Color palette (colorblind-friendly)
COLORS = {
    "AlexNet (1K classes)": "#0072B2",                        # Blue
    "ViT-B/16 (1K classes)": "#D55E00",                       # Orange
    "AlexNet Coarse\n(64 classes, AlexNet reps)": "#2E8B57",  # Sea green
    "AlexNet Coarse\n(64 classes, ViT reps)": "#66CDAA",      # Medium aquamarine
}

# Markers: AlexNet = circle, ViT = square
MARKERS = {
    "AlexNet (1K classes)": "o",
    "ViT-B/16 (1K classes)": "s",
    "AlexNet Coarse\n(64 classes, AlexNet reps)": "o",
    "AlexNet Coarse\n(64 classes, ViT reps)": "o",
}

# Line styles: 1K classes = solid, Coarse = dashed
LINESTYLES = {
    "AlexNet (1K classes)": "-",
    "ViT-B/16 (1K classes)": "-",
    "AlexNet Coarse\n(64 classes, AlexNet reps)": "--",
    "AlexNet Coarse\n(64 classes, ViT reps)": "--",
}


def load_data(csv_path):
    """Load CSV."""
    return pd.read_csv(csv_path)


def get_model_data(df, model, source, region):
    """Get filtered data for a specific model/source/region."""
    mask = (df["model"] == model) & (df["model_source"] == source) & (df["region"] == region)
    return df[mask]


def compute_mean(df):
    """Compute mean across subjects."""
    grouped = df.groupby("depth_normalized")["rsa_score"].mean()
    return grouped.index.values, grouped.values


def plot_region(ax, df, region, models_config, ylabel=True):
    """Plot RSA vs depth for a single region."""
    lines = []
    labels = []

    for model, source, display_name in models_config:
        model_df = get_model_data(df, model, source, region)

        if model_df.empty:
            print(f"Warning: No data for {display_name} in {region}")
            continue

        depths, means = compute_mean(model_df)

        # Sort by depth
        sort_idx = np.argsort(depths)
        depths = depths[sort_idx]
        means = means[sort_idx]

        color = COLORS.get(display_name, "#333333")
        marker = MARKERS.get(display_name, "o")
        linestyle = LINESTYLES.get(display_name, "-")

        line, = ax.plot(
            depths,
            means,
            color=color,
            marker=marker,
            linestyle=linestyle,
            markersize=4,
            markerfacecolor=color,
            markeredgecolor='white',
            markeredgewidth=0.4,
            linewidth=1.5,
            zorder=3
        )
        lines.append(line)
        labels.append(display_name)

    # Formatting
    ax.set_xlabel("Normalized depth")
    if ylabel:
        ax.set_ylabel("RSA score")

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(0, None)

    # Region title
    region_short = "Early Visual" if "early" in region.lower() else "Ventral Visual"
    ax.set_title(region_short, fontweight='bold', pad=4)

    return lines, labels


def main():
    # Load data
    df = load_data(INPUT_CSV)

    # Create figure with extra space for legend
    fig_width = FIGURE_WIDTH_MM * MM_TO_INCH
    fig_height = FIGURE_HEIGHT_MM * MM_TO_INCH
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, fig_height))

    plt.subplots_adjust(wspace=0.2, left=0.07, right=0.75, top=0.88, bottom=0.18)

    # Plot each region with different custom models
    lines1, labels1 = plot_region(axes[0], df, "early visual stream", MODELS_EVS, ylabel=True)
    lines2, labels2 = plot_region(axes[1], df, "ventral visual stream", MODELS_VVS, ylabel=False)

    # Common legend to the right (combine unique entries from both panels)
    all_lines = []
    all_labels = []
    seen_labels = set()

    for lines, labels in [(lines1, labels1), (lines2, labels2)]:
        for line, label in zip(lines, labels):
            if label not in seen_labels:
                all_lines.append(line)
                all_labels.append(label)
                seen_labels.add(label)

    fig.legend(
        all_lines,
        all_labels,
        loc='center right',
        bbox_to_anchor=(0.98, 0.5),
        frameon=True,
        framealpha=0.95,
        edgecolor='none',
        fancybox=False,
    )

    # Add panel labels (a, b)
    for i, ax in enumerate(axes):
        ax.text(
            -0.15, 1.08,
            chr(97 + i),
            transform=ax.transAxes,
            fontsize=10,
            fontweight='bold',
            va='top'
        )

    # Save figure
    output_path = Path(OUTPUT_FILE)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path, format='png', dpi=300)
    print(f"Saved: {output_path}")

    plt.close(fig)


if __name__ == "__main__":
    main()
