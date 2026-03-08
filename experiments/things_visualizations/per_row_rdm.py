"""
Per-row RDM correlations: CLIP 4-class vs 1K vs Behavioral.

For each THINGS concept (row in the RDM), correlates the CLIP 4-class and 1K
model RDM rows against the behavioral RDM row. The per-concept difference
reveals which concepts drive the aggregate RSA advantage.

Outputs:
  figures/per_row_scatter.png     — scatter + marginal histogram + annotated extremes
  figures/per_row_categories.png  — grouped violin by semantic category

Run from project root:
  python experiments/things_visualizations/per_row_rdm.py
"""

import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import seaborn as sns
from scipy.stats import spearmanr, wilcoxon
from matplotlib.lines import Line2D

from experiments.things_visualizations.utils import (
    load_data, FIG_DIR, save_fig, COLOR_CLIP4, COLOR_1K, COLOR_NEUTRAL,
)
from visreps.analysis.rsa import compute_rdm

sns.set_theme(style="ticks", context="paper", font_scale=1.0)

TOP_N = 10


# ── Core analysis ────────────────────────────────────────────────────

def per_row_correlations(model_rdm, behav_rdm):
    """Spearman correlation between each row of model_rdm and behav_rdm.

    Excludes diagonal (self-dissimilarity = 0) from each row.
    Returns array of shape (n_concepts,).
    """
    n = model_rdm.shape[0]
    scores = np.empty(n)
    n_nan = 0
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        rho, _ = spearmanr(model_rdm[i, mask], behav_rdm[i, mask])
        if np.isnan(rho):
            n_nan += 1
            rho = 0.0
        scores[i] = rho
    if n_nan > 0:
        print(f"  WARNING: {n_nan}/{n} rows returned NaN correlation (replaced with 0.0)")
    return scores


def assign_semantic_categories(embeddings, dimension_labels):
    """Assign each concept to its max-loading behavioral dimension."""
    max_dim = np.argmax(np.abs(embeddings), axis=1)
    return np.array([dimension_labels[d] for d in max_dim])


# ── Scatter plot with marginal histogram ─────────────────────────────

def plot_scatter(df, output_path):
    """Scatter of per-concept correlations: 1K (x) vs CLIP-4 (y), with diff histogram below."""
    fig = plt.figure(figsize=(6.0, 7.8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[2.2, 1], hspace=0.28,
                           left=0.13, right=0.95, top=0.96, bottom=0.07)
    ax_s = fig.add_subplot(gs[0])
    ax_h = fig.add_subplot(gs[1])

    # ── Refined palette ──────────────────────────────────────────────
    c_green = "#1a7a3a"
    c_red = "#c23b22"

    # ── Scatter ──────────────────────────────────────────────────────
    clip4_above = df["diff"] > 0
    for mask, color, label in [(clip4_above, c_green, "4-class > 1K"),
                                (~clip4_above, c_red, "1K > 4-class")]:
        ax_s.scatter(df.loc[mask, "corr_1k"], df.loc[mask, "corr_clip4"],
                     c=color, s=16, alpha=0.38, edgecolors="none",
                     rasterized=True, label=label, zorder=2)

    # Identity line
    lims = [min(df["corr_1k"].min(), df["corr_clip4"].min()) - 0.08,
            max(df["corr_1k"].max(), df["corr_clip4"].max()) + 0.05]
    ax_s.plot(lims, lims, color="#b0b0b0", lw=0.8, ls="--", zorder=1)
    ax_s.set_xlim(lims)
    ax_s.set_ylim(lims)

    # Annotate extreme concepts
    # Top 5 (green, upper-left region — offset left to avoid dense cloud)
    top = df.nlargest(5, "diff").reset_index(drop=True)
    top_offsets = [(-58, 10), (-60, -8), (-64, -24), (-52, -38), (-48, -50)]
    for j, (_, row) in enumerate(top.iterrows()):
        ax_s.annotate(
            row["concept"].replace("_", " "),
            (row["corr_1k"], row["corr_clip4"]),
            fontsize=6.5, color=c_green, alpha=0.85, fontstyle="italic",
            fontweight="medium",
            textcoords="offset points", xytext=top_offsets[j],
            arrowprops=dict(arrowstyle="-", color=c_green, lw=0.5, alpha=0.5,
                            shrinkA=0, shrinkB=2),
        )
    # Bottom 5 (red, lower-right region)
    # Data: buffet(0.25,-0.45), hairnet(0.32,-0.29), waffle(0.39,-0.13),
    #        fishbowl(0.12,-0.39), baby(0.26,-0.24)
    # Fan out radially from the cluster to avoid collisions
    bot = df.nsmallest(5, "diff").reset_index(drop=True)
    bot_offsets = [(14, -20), (20, -16), (18, 14), (-72, -6), (-58, 18)]
    for j, (_, row) in enumerate(bot.iterrows()):
        ax_s.annotate(
            row["concept"].replace("_", " "),
            (row["corr_1k"], row["corr_clip4"]),
            fontsize=6.5, color=c_red, alpha=0.85, fontstyle="italic",
            fontweight="medium",
            textcoords="offset points", xytext=bot_offsets[j],
            arrowprops=dict(arrowstyle="-", color=c_red, lw=0.5, alpha=0.5,
                            shrinkA=0, shrinkB=2),
        )

    ax_s.set_xlabel(r"Per-concept $\rho_s$ (1000-class vs behavioral)", fontsize=9)
    ax_s.set_ylabel(r"Per-concept $\rho_s$ (CLIP 4-class vs behavioral)", fontsize=9)
    ax_s.tick_params(labelsize=7.5, width=0.6, length=3)
    ax_s.set_aspect("equal")
    sns.despine(ax=ax_s, offset=5)
    for spine in ax_s.spines.values():
        spine.set_linewidth(0.6)

    # Legend
    leg = ax_s.legend(fontsize=7.5, frameon=True, loc="lower right",
                      markerscale=1.8, handletextpad=0.4,
                      framealpha=0.85, edgecolor="#dddddd",
                      fancybox=True, borderpad=0.5)
    leg.get_frame().set_linewidth(0.4)

    # Panel label
    ax_s.text(-0.12, 1.03, "a", transform=ax_s.transAxes,
              fontsize=13, fontweight="bold", va="top", fontfamily="sans-serif")

    # ── Histogram ────────────────────────────────────────────────────
    diff = df["diff"].values
    bins = np.linspace(diff.min() - 0.02, diff.max() + 0.02, 50)
    bin_colors = [c_green if (b_lo + b_hi) / 2 > 0 else c_red
                  for b_lo, b_hi in zip(bins[:-1], bins[1:])]
    _, _, patches = ax_h.hist(diff, bins=bins, edgecolor="white", linewidth=0.4)
    for patch, c in zip(patches, bin_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.72)

    ax_h.axvline(0, color="#aaaaaa", lw=0.7, ls="--", zorder=3)
    med = np.median(diff)
    ax_h.axvline(med, color="#222222", lw=1.2, ls="-", zorder=3)

    # Median annotation — small, offset right of the median line
    ax_h.annotate(f"Median = {med:.3f}", xy=(med, 0.96),
                  xycoords=ax_h.get_xaxis_transform(),
                  xytext=(8, 0), textcoords="offset points",
                  fontsize=6.5, va="top", ha="left", color="#444444",
                  fontstyle="italic",
                  bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

    ax_h.set_xlabel(r"$\Delta\rho_s$ (CLIP 4-class $-$ 1000-class)", fontsize=9)
    ax_h.set_ylabel("Count", fontsize=9)
    ax_h.tick_params(labelsize=7.5, width=0.6, length=3)
    sns.despine(ax=ax_h, offset=5)
    for spine in ax_h.spines.values():
        spine.set_linewidth(0.6)

    # Summary annotation — top right, below median label
    n_win = (diff > 0).sum()
    pct = 100 * n_win / len(diff)
    ax_h.text(0.98, 0.92,
              f"4-class > 1K: {n_win}/{len(diff)} ({pct:.0f}%)",
              transform=ax_h.transAxes, fontsize=7, va="top", ha="right",
              color=c_green, fontweight="bold")

    # Panel label
    ax_h.text(-0.12, 1.10, "b", transform=ax_h.transAxes,
              fontsize=13, fontweight="bold", va="top", fontfamily="sans-serif")

    fig.savefig(output_path, dpi=600, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved: {output_path}")


# ── Category enrichment plot ─────────────────────────────────────────

def _short_label(cat):
    """Shorten long dimension labels for display."""
    REMAP = {
        "music-related / hearing-related / hobby-related / loud": "Music / auditory",
        "transportation/movement-related": "Transportation / movement",
        "electronics / technology": "Electronics",
        "hobby-related / game-related / playing-related": "Games / hobbies",
        "paper-related / flat": "Paper / flat",
        "metallic/artificial": "Metallic / artificial",
        "house-related/furnishing-related": "Home / furnishing",
        "valuable/precious": "Valuable / precious",
        "circular / round": "Circular / round",
        "colorful / playful": "Colorful / playful",
        "wood-related/brown": "Wood / brown",
        "body-/people-related": "Body / people",
        "fluid-related / drink-related": "Fluid / drink",
        "body part-related": "Body parts",
    }
    return REMAP.get(cat, cat.replace("-related", "").capitalize())


def plot_categories(df, output_path, n_top=10, n_bottom=10, min_concepts=10):
    """Horizontal box + strip plot of per-concept differences by semantic category.

    Shows the top `n_top` categories where 4-class > 1K (highest median diff)
    and the bottom `n_bottom` categories where 1K > 4-class (lowest median diff).
    Only considers categories with at least `min_concepts` concepts.
    """
    cat_counts = df["category"].value_counts()
    valid_cats = cat_counts[cat_counts >= min_concepts].index
    cat_medians = (df[df["category"].isin(valid_cats)]
                   .groupby("category")["diff"].median()
                   .sort_values(ascending=False))
    top_cats = cat_medians.head(n_top).index.tolist()
    bot_cats = cat_medians.tail(n_bottom).index.tolist()
    cat_order = top_cats + bot_cats
    df_filt = df[df["category"].isin(cat_order)].copy()
    n_cats = len(cat_order)

    # Colors
    GREEN = "#2d8c2d"
    RED = "#c0392b"
    BOX_FILL = "#eaeaea"
    BOX_EDGE = "#aaaaaa"
    SHADE_COLOR = "#f2f3f5"

    fig, ax = plt.subplots(figsize=(6.0, 0.36 * n_cats + 1.2))

    # Alternating row shading (behind everything)
    for i in range(n_cats):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color=SHADE_COLOR, zorder=0, lw=0)

    # Zero reference line — dashed to distinguish from data
    ax.axvline(0, color="#888888", lw=0.7, ls="--", zorder=1.5)

    # Box plots — clean, no fliers
    box_data = [df_filt.loc[df_filt["category"] == c, "diff"].values for c in cat_order]
    bp = ax.boxplot(box_data, positions=range(n_cats), vert=False,
                    widths=0.50, patch_artist=True, showfliers=False,
                    boxprops=dict(facecolor=BOX_FILL, edgecolor=BOX_EDGE,
                                  linewidth=0.5),
                    whiskerprops=dict(color=BOX_EDGE, linewidth=0.5),
                    capprops=dict(color=BOX_EDGE, linewidth=0.5),
                    medianprops=dict(color="#1a1a1a", linewidth=1.3,
                                     solid_capstyle="round"))

    # Overlay strip (jittered points, colored by sign)
    # Scale point size inversely with category size to reduce over-plotting
    rng = np.random.default_rng(42)
    for i, cat in enumerate(cat_order):
        vals = df_filt.loc[df_filt["category"] == cat, "diff"].values
        n_pts = len(vals)
        jitter = rng.uniform(-0.17, 0.17, size=n_pts)
        colors = np.where(vals > 0, GREEN, RED)
        pt_size = 8 if n_pts < 100 else 5
        pt_alpha = 0.55 if n_pts < 100 else 0.40
        ax.scatter(vals, i + jitter, c=colors, s=pt_size, alpha=pt_alpha,
                   edgecolors="none", rasterized=True, zorder=3)

    # Y-axis labels
    short_labels = [f"{_short_label(cat)}  ($n$\u2009=\u2009{cat_counts[cat]})"
                    for cat in cat_order]
    ax.set_yticks(range(n_cats))
    ax.set_yticklabels(short_labels, fontsize=7.5)
    ax.set_ylim(n_cats - 0.5, -0.5)
    ax.set_xlabel(
        r"$\Delta\rho_s$" + " (CLIP 4-class \u2212 1000-class)",
        fontsize=9, labelpad=10)
    ax.tick_params(axis="x", labelsize=7.5, pad=4)
    ax.tick_params(axis="y", length=0, pad=8)

    # Despine
    sns.despine(ax=ax, left=True, top=True, right=True, offset=5)
    ax.spines["bottom"].set_linewidth(0.6)
    ax.spines["bottom"].set_color("#444444")

    # Legend — lower right where there's less data
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=GREEN,
               markersize=5, label="4-class > 1K", markeredgecolor="none"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=RED,
               markersize=5, label="1K > 4-class", markeredgecolor="none"),
    ]
    leg = ax.legend(handles=legend_elements, fontsize=7, frameon=True,
                    loc="lower right", handletextpad=0.4, borderpad=0.5,
                    fancybox=True, edgecolor="#e0e0e0", facecolor="white",
                    framealpha=0.95)
    leg.get_frame().set_linewidth(0.4)

    plt.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved: {output_path}")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    data = load_data()

    concept_names = [str(c) for c in data["concept_names"]]
    dimension_labels = [str(l) for l in data["dimension_labels"]]
    n_concepts = len(concept_names)
    print(f"Loaded {n_concepts} concepts, building RDMs...")

    # Build full RDMs (Pearson-based dissimilarity)
    rdm_behav = compute_rdm(torch.tensor(data["embeddings"], dtype=torch.float32)).numpy()
    rdm_clip4 = compute_rdm(torch.tensor(data["clip4_acts"], dtype=torch.float32)).numpy()
    rdm_1k = compute_rdm(torch.tensor(data["thousand_acts"], dtype=torch.float32)).numpy()
    print(f"RDMs: {rdm_behav.shape}")

    # Per-row correlations
    print("Computing per-row Spearman correlations...")
    corr_clip4 = per_row_correlations(rdm_clip4, rdm_behav)
    corr_1k = per_row_correlations(rdm_1k, rdm_behav)
    diff = corr_clip4 - corr_1k

    # Assign semantic categories from max-loading behavioral dimension
    categories = assign_semantic_categories(data["embeddings"], dimension_labels)

    df = pd.DataFrame({
        "concept": concept_names,
        "corr_clip4": corr_clip4,
        "corr_1k": corr_1k,
        "diff": diff,
        "category": categories,
    })

    # ── Summary stats ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Per-concept RDM row correlations (n={n_concepts})")
    print(f"{'='*60}")
    print(f"  CLIP 4-class mean rho:  {corr_clip4.mean():.4f} +/- {corr_clip4.std():.4f}")
    print(f"  1K mean rho:            {corr_1k.mean():.4f} +/- {corr_1k.std():.4f}")
    print(f"  Mean diff (CLIP4 - 1K): {diff.mean():.4f}")
    print(f"  Median diff:            {np.median(diff):.4f}")
    print(f"  CLIP-4 wins:            {(diff > 0).sum()}/{n_concepts} ({100*(diff>0).mean():.1f}%)")

    # Wilcoxon signed-rank test
    stat, pval = wilcoxon(corr_clip4, corr_1k)
    print(f"\n  Wilcoxon signed-rank: W={stat:.0f}, p={pval:.2e}")

    # Top/bottom concepts
    print(f"\n  Top {TOP_N} concepts (CLIP-4 advantage):")
    for _, row in df.nlargest(TOP_N, "diff").iterrows():
        print(f"    {row['concept']:<22} diff={row['diff']:+.4f}  "
              f"(clip4={row['corr_clip4']:.3f}, 1k={row['corr_1k']:.3f})")

    print(f"\n  Top {TOP_N} concepts (1K advantage):")
    for _, row in df.nsmallest(TOP_N, "diff").iterrows():
        print(f"    {row['concept']:<22} diff={row['diff']:+.4f}  "
              f"(clip4={row['corr_clip4']:.3f}, 1k={row['corr_1k']:.3f})")

    # ── Plots ────────────────────────────────────────────────────
    plot_scatter(df, os.path.join(FIG_DIR, "per_row_scatter.png"))
    plot_categories(df, os.path.join(FIG_DIR, "per_row_categories.png"))

    print("\nDone!")


if __name__ == "__main__":
    main()
