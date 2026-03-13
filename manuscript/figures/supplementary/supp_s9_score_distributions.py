"""Supplementary Figure S9: Score Distributions Across Subjects and Seeds.

Violin + strip plots showing the full spread of individual data points
(across all subjects x seeds) for each coarseness level, per dataset and ROI.

Layout: 2 rows x 3 columns
  Row 1: TVSD V1, TVSD V4, TVSD IT
  Row 2: NSD Early Visual, NSD Ventral Visual, THINGS

Each panel auto-selects the best PCA architecture for that (dataset, region).

Usage:
    python manuscript/figures/supplementary/supp_s9_score_distributions.py
"""

import sqlite3
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.insert(0, ".")
from manuscript.figures.fig_utils import (
    COARSE_CFGS,
    GRAN_COLORS,
    find_best_architecture,
    setup_style,
)

OUTPUT = "manuscript/figures/supplementary/supp_s9_score_distributions.png"
DB_PATH = "results.db"

UNTRAINED_COLOR = "#AAAAAA"
LEVELS = COARSE_CFGS + [1000]
LEVEL_LABELS = [str(c) for c in LEVELS] + ["Untrained"]

# Panel definitions: (neural_dataset, region_key, display_title)
PANELS = [
    # Row 0
    ("tvsd", "V1", "TVSD V1"),
    ("tvsd", "V4", "TVSD V4"),
    ("tvsd", "IT", "TVSD IT"),
    # Row 1
    ("nsd", "early visual stream", "NSD Early Visual"),
    ("nsd", "ventral visual stream", "NSD Ventral Visual"),
    ("things-behavior", "N/A", "THINGS"),
]


def get_all_scores(neural_dataset, region, pca_labels_folder, cfg_id, epoch=20):
    """Get all individual (subject x seed) scores for a condition."""
    conn = sqlite3.connect(DB_PATH)
    if cfg_id == 1000:
        folder_cond = "pca_labels_folder = 'imagenet1k'"
    else:
        folder_cond = f"pca_labels_folder = '{pca_labels_folder}'"

    query = f"""
        SELECT score, seed, subject_idx
        FROM results
        WHERE neural_dataset = ? AND region = ? AND cfg_id = ?
          AND compare_method = 'spearman' AND reconstruct_from_pcs = 0
          AND analysis = 'rsa' AND epoch = ?
          AND {folder_cond}
    """
    df = pd.read_sql(query, conn, params=[neural_dataset, region, cfg_id, epoch])
    conn.close()
    return df


def collect_panel_data(neural_dataset, region, pca_folder):
    """Collect all individual scores for all granularity levels + untrained."""
    rows = []
    for cfg in LEVELS:
        df = get_all_scores(neural_dataset, region, pca_folder, cfg, epoch=20)
        for _, row in df.iterrows():
            rows.append({"classes": str(cfg), "score": row["score"]})

    # Untrained (epoch=0, cfg_id=1000)
    df = get_all_scores(neural_dataset, region, pca_folder, 1000, epoch=0)
    for _, row in df.iterrows():
        rows.append({"classes": "Untrained", "score": row["score"]})

    return pd.DataFrame(rows)


def make_palette():
    """Build color palette matching GRAN_COLORS + untrained gray."""
    palette = {str(cfg): GRAN_COLORS[cfg] for cfg in LEVELS}
    palette["Untrained"] = UNTRAINED_COLOR
    return palette


def main():
    setup_style()
    palette = make_palette()
    order = [str(c) for c in LEVELS] + ["Untrained"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5))

    for panel_idx, (neural_dataset, region, display_title) in enumerate(PANELS):
        row, col = divmod(panel_idx, 3)
        ax = axes[row, col]

        # Auto-select best architecture
        pca_folder, arch_display = find_best_architecture(neural_dataset, region)
        print(f"  [{display_title}]: best architecture = {arch_display} ({pca_folder})")

        df = collect_panel_data(neural_dataset, region, pca_folder)

        if df.empty:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=10, color="#888")
            continue

        # Count points per condition to decide violin vs box
        n_per_condition = df.groupby("classes").size().min()
        use_violin = n_per_condition >= 6

        if use_violin:
            # Violin plot
            sns.violinplot(
                data=df, x="classes", y="score", hue="classes",
                order=order, hue_order=order,
                palette=palette, ax=ax, inner=None, linewidth=0.7,
                cut=0, density_norm="width", width=0.7,
                saturation=0.8, legend=False,
            )
            # Make violins semi-transparent
            for collection in ax.collections:
                collection.set_alpha(0.35)
        else:
            # Box plot for small sample sizes
            sns.boxplot(
                data=df, x="classes", y="score", hue="classes",
                order=order, hue_order=order,
                palette=palette, ax=ax, width=0.5,
                fliersize=0, linewidth=0.8,
                boxprops=dict(facecolor="white", alpha=0.5),
                medianprops=dict(color="black", linewidth=1.2),
                whiskerprops=dict(linewidth=0.8),
                capprops=dict(linewidth=0.8),
                legend=False,
            )

        # Strip plot overlay (individual points)
        sns.stripplot(
            data=df, x="classes", y="score", hue="classes",
            order=order, hue_order=order,
            palette=palette, ax=ax, size=3.5 if use_violin else 4.5,
            alpha=0.8, jitter=0.12, edgecolor="white", linewidth=0.4,
            zorder=3, legend=False,
        )

        # Title with architecture info
        ax.set_title(f"{display_title}\n({arch_display})",
                     fontsize=9, fontweight="semibold", pad=6)

        # Axis formatting
        if row == 1:
            ax.set_xlabel("Classes", fontsize=8, labelpad=4)
        else:
            ax.set_xlabel("")
        if col == 0:
            ax.set_ylabel(r"Spearman $\rho$", fontsize=9, labelpad=4)
        else:
            ax.set_ylabel("")

        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.yaxis.grid(True, which="major", color="#EBEBEB", linewidth=0.4, zorder=0)
        sns.despine(ax=ax, right=True, top=True, offset=4)

    # Panel labels
    for i, ax in enumerate(axes.flat):
        label = chr(ord("A") + i)
        ax.text(-0.06, 1.10, label, transform=ax.transAxes,
                fontsize=13, fontweight="bold", va="top", ha="left")

    fig.tight_layout(h_pad=3.0, w_pad=2.0)
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print(f"\nSaved -> {OUTPUT}")
    plt.close()


if __name__ == "__main__":
    main()
