"""
Per-row RDM scatter colored by THINGS 27 manual categories.

Shows per-concept RDM-row correlations: 1K (x) vs CLIP 4-class (y).
Points are colored by their THINGS category:
  - Top 5 categories with positive median Δρ → green palette (4-class advantage)
  - All categories with negative median Δρ → orange palette (1K advantage)
  - Remaining categories + buffer zone → grey

Each highlighted category gets a distinct color + marker shape.

Run from project root:
  python experiments/things_visualizations/per_row_scatter_categories.py
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib as mpl
import seaborn as sns
from scipy.stats import spearmanr
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from experiments.things_visualizations.utils import load_data, FIG_DIR
from visreps.analysis.rsa import compute_rdm

sns.set_theme(style="ticks", context="paper", font_scale=1.0)
mpl.rcParams.update({
    "font.family": "sans-serif",
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
})

CATEGORY_FILE = os.path.expanduser(
    "~/.cache/bonner-datasets/hebart2019.things/03_category-level/category27_manual.tsv"
)

N_TOP_GREEN = 5  # top positive-median categories to highlight

# Distinct colors within green-ish and orange-ish families
GREEN_COLORS = ["#0b6623", "#2e86ab", "#52b788", "#8ecae6", "#95d5b2"]
GREEN_MARKERS = ["o", "s", "^", "D", "v"]

ORANGE_COLORS = ["#c1121f", "#e07a28", "#d4a373", "#f4a261"]
ORANGE_MARKERS = ["o", "s", "^", "D"]

GREY = "#cdcdcd"
GREY_BUFFER = "#d9d9d9"


# ── Data loading & analysis ──────────────────────────────────────────

def per_row_correlations(model_rdm, behav_rdm):
    n = model_rdm.shape[0]
    scores = np.empty(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        rho, _ = spearmanr(model_rdm[i, mask], behav_rdm[i, mask])
        scores[i] = 0.0 if np.isnan(rho) else rho
    return scores


def load_categories():
    cat_df = pd.read_csv(CATEGORY_FILE, sep="\t")
    cat_names = list(cat_df.columns)
    categories = []
    for _, row in cat_df.iterrows():
        assigned = [c for c in cat_names if row[c] == 1]
        categories.append(assigned[0] if assigned else "Other")
    return np.array(categories)


def short_cat_label(cat):
    REMAP = {
        "body part": "Body part",
        "clothing accessory": "Clothing acc.",
        "electronic device": "Electronic",
        "home decor": "Home decor",
        "kitchen appliance": "Kitchen appl.",
        "kitchen tool": "Kitchen tool",
        "medical equipment": "Medical equip.",
        "musical instrument": "Music instr.",
        "office supply": "Office supply",
        "part of car": "Car part",
        "sports equipment": "Sports equip.",
    }
    return REMAP.get(cat, cat.capitalize())


# ── Scatter plot ─────────────────────────────────────────────────────

def plot_scatter(df, cat_medians, buffer_threshold, output_path):
    """Scatter with category coloring, buffer zone, and marginal histogram."""

    # Top 5 positive-median categories (4-class advantage)
    positive_cats = cat_medians[cat_medians > 0]
    top_cats = positive_cats.head(N_TOP_GREEN).index.tolist()

    # Negative-median categories with |median| >= 0.05 (1K advantage)
    # sorted most negative first
    negative_cats = cat_medians[cat_medians < -0.05].sort_values(ascending=True)
    bot_cats = negative_cats.index.tolist()
    n_bot = len(bot_cats)

    green_colors = GREEN_COLORS[:len(top_cats)]
    green_markers = GREEN_MARKERS[:len(top_cats)]
    orange_colors = ORANGE_COLORS[:n_bot]
    orange_markers = ORANGE_MARKERS[:n_bot]

    # Map category → (color, marker)
    cat_style = {}
    for i, cat in enumerate(top_cats):
        cat_style[cat] = (green_colors[i], green_markers[i])
    for i, cat in enumerate(bot_cats):
        cat_style[cat] = (orange_colors[i], orange_markers[i])

    # Assign per-concept properties
    df = df.copy()
    colors, markers, is_highlighted = [], [], []
    for _, row in df.iterrows():
        if abs(row["diff"]) < buffer_threshold:
            colors.append(GREY_BUFFER)
            markers.append("o")
            is_highlighted.append(False)
        elif row["category"] in cat_style:
            c, m = cat_style[row["category"]]
            colors.append(c)
            markers.append(m)
            is_highlighted.append(True)
        else:
            colors.append(GREY)
            markers.append("o")
            is_highlighted.append(False)
    df["color"] = colors
    df["marker"] = markers
    df["highlighted"] = is_highlighted

    # ── Figure layout ────────────────────────────────────────────────
    fig = plt.figure(figsize=(5.5, 7.4))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.5, 1], hspace=0.25,
                           left=0.14, right=0.96, top=0.97, bottom=0.07)
    ax_s = fig.add_subplot(gs[0])
    ax_h = fig.add_subplot(gs[1])

    # ── Buffer zone shading ──────────────────────────────────────────
    lims = [min(df["corr_1k"].min(), df["corr_clip4"].min()) - 0.08,
            max(df["corr_1k"].max(), df["corr_clip4"].max()) + 0.05]
    xx = np.linspace(lims[0], lims[1], 200)
    ax_s.fill_between(xx, xx - buffer_threshold, xx + buffer_threshold,
                       color="#f4f4f4", alpha=0.8, zorder=0, lw=0)
    ax_s.plot(xx, xx + buffer_threshold, color="#d0d0d0", lw=0.4, ls=":",
              zorder=0.3, alpha=0.7)
    ax_s.plot(xx, xx - buffer_threshold, color="#d0d0d0", lw=0.4, ls=":",
              zorder=0.3, alpha=0.7)

    # Identity line
    ax_s.plot(lims, lims, color="#b0b0b0", lw=0.6, ls="--", zorder=0.5)

    # Buffer zone label
    buf_x = -0.15
    buf_y = buf_x + buffer_threshold * 0.6
    ax_s.text(buf_x, buf_y, r"$\approx$ equal", fontsize=5.5, color="#a0a0a0",
              rotation=45, ha="center", va="center", fontstyle="italic",
              zorder=0.6)

    # ── Scatter ──────────────────────────────────────────────────────
    # Layer 1: Grey points (background)
    grey_df = df[~df["highlighted"]]
    ax_s.scatter(
        grey_df["corr_1k"], grey_df["corr_clip4"],
        c=grey_df["color"].values, s=6, marker="o",
        alpha=0.18, edgecolors="none", rasterized=True, zorder=1,
    )

    # Layer 2: Colored points — one scatter call per category for distinct markers
    for cat in top_cats + bot_cats:
        cat_mask = (df["category"] == cat) & df["highlighted"]
        subset = df[cat_mask]
        if subset.empty:
            continue
        c, m = cat_style[cat]
        ax_s.scatter(
            subset["corr_1k"], subset["corr_clip4"],
            c=c, s=14, marker=m,
            alpha=0.65, edgecolors="white", linewidths=0.15,
            rasterized=True, zorder=2,
        )

    ax_s.set_xlim(lims)
    ax_s.set_ylim(lims)
    ax_s.set_xlabel(r"Per-concept $\rho_s$ (1000-class vs behavioral)")
    ax_s.set_ylabel(r"Per-concept $\rho_s$ (CLIP 4-class vs behavioral)")
    ax_s.set_aspect("equal")
    sns.despine(ax=ax_s, offset=4)

    # ── Cluster text labels (no arrows) ─────────────────────────────
    # Compute mean positions then manually offset to avoid overlaps
    def _cat_center(cat_name):
        d = df[(df["category"] == cat_name) & df["highlighted"]]
        return d["corr_1k"].mean(), d["corr_clip4"].mean() if len(d) else (0, 0)

    annot_list = []
    if len(top_cats) >= 2:
        # Plant: place label above-left of cluster
        annot_list.append((top_cats[0], green_colors[0], (-0.10, +0.04)))
        # Animal: place label left of cluster, below plant
        annot_list.append((top_cats[1], green_colors[1], (-0.12, -0.02)))
    if n_bot >= 1:
        # Body part (most negative): place label below cluster
        annot_list.append((bot_cats[0], orange_colors[0], (-0.02, -0.06)))
    if n_bot >= 2:
        # Drink: place label right of cluster
        annot_list.append((bot_cats[1], orange_colors[1], (+0.06, -0.02)))

    for cat, color, (dx, dy) in annot_list:
        cat_data = df[(df["category"] == cat) & df["highlighted"]]
        if cat_data.empty:
            continue
        cx, cy = cat_data["corr_1k"].mean(), cat_data["corr_clip4"].mean()
        ax_s.text(
            cx + dx, cy + dy,
            short_cat_label(cat),
            fontsize=6.5, color=color,
            fontweight="bold", fontstyle="italic", alpha=0.9,
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                      alpha=0.82),
            zorder=5,
        )

    # ── Legend ────────────────────────────────────────────────────────
    legend_elements = []

    # Green header
    legend_elements.append(
        Line2D([0], [0], marker="None", color="w", label="4-class advantage")
    )
    for i, cat in enumerate(top_cats):
        med = cat_medians[cat]
        n_cat = (df["category"] == cat).sum()
        legend_elements.append(
            Line2D([0], [0], marker=green_markers[i], color="w",
                   markerfacecolor=green_colors[i], markersize=5,
                   markeredgecolor="none",
                   label=f"  {short_cat_label(cat)}  (med={med:+.2f}, n={n_cat})")
        )

    # Spacer
    legend_elements.append(
        Line2D([0], [0], marker="None", color="w", label=" ")
    )

    # Orange header
    legend_elements.append(
        Line2D([0], [0], marker="None", color="w", label="1K advantage")
    )
    for i, cat in enumerate(bot_cats):
        med = cat_medians[cat]
        n_cat = (df["category"] == cat).sum()
        legend_elements.append(
            Line2D([0], [0], marker=orange_markers[i], color="w",
                   markerfacecolor=orange_colors[i], markersize=5,
                   markeredgecolor="none",
                   label=f"  {short_cat_label(cat)}  (med={med:+.2f}, n={n_cat})")
        )

    leg = ax_s.legend(
        handles=legend_elements, fontsize=5.5, frameon=True,
        loc="lower right", markerscale=1.1, handletextpad=0.3,
        framealpha=0.95, edgecolor="#e0e0e0", fancybox=True,
        borderpad=0.5, labelspacing=0.22,
    )
    leg.get_frame().set_linewidth(0.3)
    for text in leg.get_texts():
        label = text.get_text()
        if label in ("4-class advantage", "1K advantage"):
            text.set_fontweight("bold")
            text.set_fontsize(6)

    # Panel label
    ax_s.text(-0.11, 1.02, "a", transform=ax_s.transAxes,
              fontsize=12, fontweight="bold", va="top")

    # ── Histogram ────────────────────────────────────────────────────
    diff = df["diff"].values
    bins = np.linspace(diff.min() - 0.02, diff.max() + 0.02, 48)

    c_green_hist = "#1a8a42"
    c_orange_hist = "#d95e1a"
    bin_colors = [c_green_hist if (b_lo + b_hi) / 2 > 0 else c_orange_hist
                  for b_lo, b_hi in zip(bins[:-1], bins[1:])]
    _, _, patches = ax_h.hist(diff, bins=bins, edgecolor="white", linewidth=0.3)
    for patch, c in zip(patches, bin_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.70)

    # Buffer zone shading in histogram
    ax_h.axvspan(-buffer_threshold, buffer_threshold,
                 color="#f4f4f4", alpha=0.7, zorder=0, lw=0)

    ax_h.axvline(0, color="#b0b0b0", lw=0.5, ls="--", zorder=3)
    med_val = np.median(diff)
    ax_h.axvline(med_val, color="#333333", lw=1.0, ls="-", zorder=3)

    ax_h.annotate(f"Median = {med_val:.3f}", xy=(med_val, 0.96),
                  xycoords=ax_h.get_xaxis_transform(),
                  xytext=(8, 0), textcoords="offset points",
                  fontsize=6, va="top", ha="left", color="#555555",
                  fontstyle="italic",
                  bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none",
                            alpha=0.88))

    ax_h.set_xlabel(r"$\Delta\rho_s$ (CLIP 4-class $-$ 1000-class)")
    ax_h.set_ylabel("Count")
    sns.despine(ax=ax_h, offset=4)

    n_win = (diff > 0).sum()
    pct = 100 * n_win / len(diff)
    ax_h.text(0.97, 0.92,
              f"4-class > 1K: {n_win}/{len(diff)} ({pct:.0f}%)",
              transform=ax_h.transAxes, fontsize=6.5, va="top", ha="right",
              color=c_green_hist, fontweight="bold")

    ax_h.text(-0.11, 1.08, "b", transform=ax_h.transAxes,
              fontsize=12, fontweight="bold", va="top")

    fig.savefig(output_path, dpi=600, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved: {output_path}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    data = load_data()

    concept_names = [str(c) for c in data["concept_names"]]
    n_concepts = len(concept_names)
    print(f"Loaded {n_concepts} concepts, building RDMs...")

    rdm_behav = compute_rdm(torch.tensor(data["embeddings"], dtype=torch.float32)).numpy()
    rdm_clip4 = compute_rdm(torch.tensor(data["clip4_acts"], dtype=torch.float32)).numpy()
    rdm_1k = compute_rdm(torch.tensor(data["thousand_acts"], dtype=torch.float32)).numpy()

    print("Computing per-row Spearman correlations...")
    corr_clip4 = per_row_correlations(rdm_clip4, rdm_behav)
    corr_1k = per_row_correlations(rdm_1k, rdm_behav)
    diff = corr_clip4 - corr_1k

    categories = load_categories()

    df = pd.DataFrame({
        "concept": concept_names,
        "corr_clip4": corr_clip4,
        "corr_1k": corr_1k,
        "diff": diff,
        "category": categories,
    })

    # Category-level medians (exclude Other, no minimum size restriction)
    cat_medians = (df[df["category"] != "Other"]
                   .groupby("category")["diff"].median()
                   .sort_values(ascending=False))

    print("\nCategory medians (excl. Other):")
    for cat, med in cat_medians.items():
        n = (categories == cat).sum()
        marker = "  <-- 1K" if med < 0 else ""
        print(f"  {cat:<25} {med:+.4f}  (n={n}){marker}")

    n_neg = (cat_medians < 0).sum()
    print(f"\n  {n_neg} categories with negative median (1K advantage)")

    # Generate with buffer=0.05
    plot_scatter(df, cat_medians, 0.05, os.path.join(FIG_DIR, "per_row_scatter_categories.png"))

    print("\nDone!")


if __name__ == "__main__":
    main()
