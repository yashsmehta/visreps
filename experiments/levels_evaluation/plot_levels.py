"""
Visualize Levels evaluation results: accuracy and uncertainty alignment
across coarseness levels and PCA label sources.

Usage (from project root):
    python experiments/levels_evaluation/plot_levels.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path

sns.set_theme(style="ticks", context="paper", font_scale=1.05)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.linewidth": 1.0,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
})

OUTPUT_DIR = Path(__file__).resolve().parent
SOURCES = ["alexnet", "clip", "vit"]
SOURCE_LABELS = {"alexnet": "AlexNet", "clip": "CLIP", "dino": "DINO", "vit": "ViT"}
# Match manuscript/figures/plot_coarseness_log.py color scheme
COLORS = {"alexnet": "#2166AC", "clip": "#1B7837", "dino": "#E08214", "vit": "#C51B7D"}
MARKERS = {"alexnet": "o", "clip": "s", "dino": "D", "vit": "^"}
EDGE_COLOR = "#333333"
EDGE_WIDTH = 0.5
BASELINE_COLOR = "#404040"
COARSENESS = [2, 4, 8, 16, 32, 64]


def load_summary():
    df = pd.read_csv(OUTPUT_DIR / "levels_summary.csv")
    # Parse model name into source and cfg_id
    records = []
    for _, row in df.iterrows():
        model = row["model"]
        if model == "cfg1000":
            records.append({**row, "source": "fine", "cfg_id": 1000})
        else:
            # e.g. "cfg8_clip" -> cfg_id=8, source="clip"
            parts = model.split("_", 1)
            cfg_id = int(parts[0].replace("cfg", ""))
            source = parts[1]
            records.append({**row, "source": source, "cfg_id": cfg_id})
    return pd.DataFrame(records)



def plot_by_triplet_type(df):
    """1x3 grid per metric: finest to coarsest triplet type (no overall)."""
    fine = df[df["source"] == "fine"]
    # Ordered finest -> coarsest
    triplet_types = ["within_class", "class_border", "between_class"]
    titles = ["Within-class (finest)", "Class-boundary", "Between-class (coarsest)"]

    for metric, metric_label, fname in [
        ("accuracy", "Odd-one-out accuracy", "levels_by_triplet_accuracy.png"),
        ("uncertainty_r", "Uncertainty alignment\n(Spearman $r$)", "levels_by_triplet_uncertainty.png"),
        ("rsa_r", "Triplet RSA\n(Spearman $r$)", "levels_by_triplet_rsa.png"),
    ]:
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=False)
        for ax, tt, title in zip(axes, triplet_types, titles):
            sub_all = df[df["triplet_type"] == tt]
            fine_val = fine[fine["triplet_type"] == tt].iloc[0][metric]

            for src in SOURCES:
                sub = sub_all[sub_all["source"] == src].sort_values("cfg_id")
                ax.plot(sub["cfg_id"], sub[metric], marker=MARKERS[src],
                        color=COLORS[src], label=SOURCE_LABELS[src],
                        markersize=6, linewidth=1.5,
                        markeredgecolor=EDGE_COLOR,
                        markeredgewidth=EDGE_WIDTH,
                        clip_on=False, zorder=3)

            ax.axhline(fine_val, color=BASELINE_COLOR, linestyle="--",
                       linewidth=1.1, label="1000-class", zorder=1)
            ax.set_title(title, fontweight="semibold", pad=8, fontsize=11)
            ax.set_xscale("log", base=2)
            ax.set_xticks(COARSENESS)
            ax.set_xticklabels(COARSENESS)
            ax.xaxis.set_minor_locator(mticker.NullLocator())
            ax.set_ylabel(metric_label)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))

            # Light horizontal gridlines
            ax.grid(axis="y", color="#EBEBEB", linewidth=0.4, zorder=0)
            ax.set_axisbelow(True)

            # Pad y-limits
            ymin, ymax = ax.get_ylim()
            margin = (ymax - ymin) * 0.08
            ax.set_ylim(ymin - margin, ymax + margin)

            sns.despine(ax=ax, offset=5)

        # Only label y-axis on leftmost panel; shared x-label
        for ax in axes[1:]:
            ax.set_ylabel("")

        # Single legend above, centered
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=5,
                   frameon=False, bbox_to_anchor=(0.5, 1.05),
                   columnspacing=1.8, handletextpad=0.5,
                   markerscale=1.2)

        fig.tight_layout()
        fig.supxlabel("Number of classes", fontsize=11, fontweight="medium", y=-0.02)
        fig.savefig(OUTPUT_DIR / fname, dpi=600, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        print(f"Saved {OUTPUT_DIR / fname}")
        plt.close(fig)


if __name__ == "__main__":
    df = load_summary()
    plot_by_triplet_type(df)
