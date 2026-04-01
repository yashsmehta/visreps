"""
Supplementary Figure S16: Levels Evaluation (Muttenthaler et al. 2025).

Shows results on the Levels dataset -- a hierarchical similarity benchmark.
1x3 layout per metric, split by triplet type (within_class, class_border,
between_class). Plots all three metrics in a 3x3 grid.

Adapted from experiments/levels_evaluation/plot_levels.py.

Run from project root:
    python manuscript/figures/supplementary/supp_s16_levels.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from manuscript.figures.fig_utils import setup_style

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "supp_s16_levels.png")
DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "experiments", "levels_evaluation", "levels_summary.csv"
)

SOURCES = ["alexnet", "clip", "dino", "vit"]
SOURCE_LABELS = {"alexnet": "AlexNet", "clip": "CLIP", "dino": "DINO", "vit": "ViT"}
COLORS = {"alexnet": "#1a9e76", "clip": "#7b3294", "dino": "#17becf", "vit": "#d62728"}
MARKERS = {"alexnet": "o", "clip": "s", "dino": "p", "vit": "^"}
BASELINE_COLOR = "#404040"
COARSENESS = [2, 4, 8, 16, 32, 64]

TRIPLET_TYPES = ["within_class", "class_border", "between_class"]
TRIPLET_TITLES = ["Within-class", "Class-boundary", "Between-class"]

METRICS = [
    ("accuracy", "Odd-one-out\naccuracy"),
    ("uncertainty_r", "Uncertainty alignment\n(Spearman $r$)"),
    ("rsa_r", "Triplet RSA\n(Spearman $r$)"),
]


def load_summary():
    df = pd.read_csv(DATA_PATH)
    records = []
    for _, row in df.iterrows():
        model = row["model"]
        if model == "cfg1000":
            records.append({**row, "source": "fine", "cfg_id": 1000})
        else:
            parts = model.split("_", 1)
            cfg_id = int(parts[0].replace("cfg", ""))
            source = parts[1]
            records.append({**row, "source": source, "cfg_id": cfg_id})
    return pd.DataFrame(records)


def main():
    setup_style()

    print("Loading Levels summary data...")
    df = load_summary()
    fine = df[df["source"] == "fine"]

    fig, axes = plt.subplots(3, 3, figsize=(11, 9.5), sharey=False)

    for row_idx, (metric, metric_label) in enumerate(METRICS):
        for col_idx, (tt, tt_title) in enumerate(zip(TRIPLET_TYPES, TRIPLET_TITLES)):
            ax = axes[row_idx, col_idx]
            sub_all = df[df["triplet_type"] == tt]
            fine_val = fine[fine["triplet_type"] == tt].iloc[0][metric]

            for src in SOURCES:
                sub = sub_all[sub_all["source"] == src].sort_values("cfg_id")
                ax.plot(sub["cfg_id"], sub[metric],
                        marker=MARKERS[src], color=COLORS[src],
                        label=SOURCE_LABELS[src],
                        markersize=5.5, linewidth=1.4,
                        markeredgecolor="white", markeredgewidth=0.5,
                        clip_on=False, zorder=3)

            ax.axhline(fine_val, color=BASELINE_COLOR, linestyle="--",
                       linewidth=1.0, label="1000-class", zorder=1)

            # Column titles only on top row
            if row_idx == 0:
                ax.set_title(tt_title, fontweight="semibold", pad=8, fontsize=10)

            ax.set_xscale("log", base=2)
            ax.set_xticks(COARSENESS)
            ax.set_xticklabels(COARSENESS)
            ax.xaxis.set_minor_locator(mticker.NullLocator())

            # Y-axis label only on leftmost column
            if col_idx == 0:
                ax.set_ylabel(metric_label, fontsize=9)
            else:
                ax.set_ylabel("")

            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
            ax.grid(axis="y", color="#EBEBEB", linewidth=0.4, zorder=0)
            ax.set_axisbelow(True)

            # Pad y-limits
            ymin, ymax = ax.get_ylim()
            margin = (ymax - ymin) * 0.08
            ax.set_ylim(ymin - margin, ymax + margin)

            # X-axis label only on bottom row
            if row_idx == 2:
                ax.set_xlabel("Number of classes", fontsize=9)

            sns.despine(ax=ax, offset=5)

    # Single legend above, centered
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5,
               frameon=False, bbox_to_anchor=(0.5, 1.02),
               columnspacing=1.8, handletextpad=0.5,
               markerscale=1.2, fontsize=9)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
