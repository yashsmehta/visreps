"""
Bar plot of brain alignment across all coarseness levels (untrained, 2–64, 1000).

Bar height = grand mean (over subjects and seeds).
Error bars = SD across 3 seeds (each seed averaged over 8 subjects).
Significance stars on coarse bars that exceed the 1000-class baseline
(paired t-test, seed-averaged, paired by subject, N=8).

Includes untrained (epoch=0) reference bar in gray and an axis break
between 64 and 1000 to reflect the non-linear x-axis spacing.

Styling inspired by experiments/neurips_2025/fig2/bar_plot_nsd.py.

Run with:  python figures/coarseness_barplot.py --folder pca_labels_alexnet
"""

import argparse
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import seaborn as sns
from scipy import stats

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--folder", required=True,
                    choices=["pca_labels_alexnet", "pca_labels_vit",
                             "pca_labels_clip", "pca_labels_dino"])
args = parser.parse_args()

PCA_FOLDER = args.folder
FOLDER_DISPLAY = {
    "pca_labels_alexnet": "AlexNet",
    "pca_labels_vit": "ViT",
    "pca_labels_clip": "CLIP",
    "pca_labels_dino": "DINO",
}
DISPLAY_NAME = FOLDER_DISPLAY[PCA_FOLDER]

# ── Style ────────────────────────────────────────────────────────────────────
sns.set_theme(style="ticks", context="paper", font_scale=1.1)
plt.rcParams["hatch.color"] = "grey"

COARSE_CFGS = [2, 4, 8, 16, 32, 64]
N_COARSE = len(COARSE_CFGS)

# Gradient of blues for coarse levels (light → dark)
blues = sns.color_palette("Blues", n_colors=N_COARSE + 1)[1:]
UNTRAINED_COLOR = "#AAAAAA"
BASELINE_COLOR = "#FFA500"

# X positions: untrained | small gap | coarse bars | break gap | 1000
X_UNTRAINED = 0
X_COARSE = np.arange(1.5, 1.5 + N_COARSE)
X_BASELINE = X_COARSE[-1] + 2
ALL_X = np.array([X_UNTRAINED] + list(X_COARSE) + [X_BASELINE])

ALL_COLORS = [UNTRAINED_COLOR] + list(blues) + [BASELINE_COLOR]
ALL_HATCHES = [""] + ["/"] * N_COARSE + [""]  # hatches on coarse bars only
ALL_LABELS = ["Untrained"] + [str(c) for c in COARSE_CFGS] + ["1000"]

BAR_WIDTH = 0.72
BREAK_X = (X_COARSE[-1] + X_BASELINE) / 2

# ── Data ─────────────────────────────────────────────────────────────────────
DB_PATH = "results.db"

QUERY_TRAINED = f"""
SELECT compare_method, score, seed, region, subject_idx, cfg_id, pca_labels_folder, epoch
FROM results
WHERE (pca_labels_folder = '{PCA_FOLDER}' OR pca_labels_folder = 'imagenet1k')
  AND compare_method IN ('spearman', 'kendall')
  AND epoch = 20
"""

QUERY_UNTRAINED = """
SELECT compare_method, score, seed, region, subject_idx, cfg_id, pca_labels_folder, epoch
FROM results
WHERE epoch = 0
  AND compare_method IN ('spearman', 'kendall')
"""

print(f"SQL (trained):\n{QUERY_TRAINED}")
print(f"SQL (untrained):\n{QUERY_UNTRAINED}")

conn = sqlite3.connect(DB_PATH)
df_trained = pd.read_sql(QUERY_TRAINED, conn)
df_untrained = pd.read_sql(QUERY_UNTRAINED, conn)
conn.close()

df = pd.concat([df_trained, df_untrained], ignore_index=True)

# ── Precompute ───────────────────────────────────────────────────────────────
regions = ["early visual stream", "ventral visual stream"]
metrics = ["spearman", "kendall"]
region_labels = {"early visual stream": "Early Visual Stream",
                 "ventral visual stream": "Ventral Visual Stream"}
metric_labels = {"spearman": "Spearman ρ", "kendall": "Kendall τ"}

df_seed_avg = (
    df.groupby(["compare_method", "region", "seed", "cfg_id", "pca_labels_folder", "epoch"])["score"]
    .mean()
    .reset_index()
)

df_subj_avg = (
    df.groupby(["compare_method", "region", "subject_idx", "cfg_id", "pca_labels_folder", "epoch"])["score"]
    .mean()
    .reset_index()
)


def get_bar_stats(metric, region, cfg_id, folder, epoch):
    """Return (mean, sd) across seeds for a condition."""
    mask = (
        (df_seed_avg["compare_method"] == metric)
        & (df_seed_avg["region"] == region)
        & (df_seed_avg["cfg_id"] == cfg_id)
        & (df_seed_avg["pca_labels_folder"] == folder)
        & (df_seed_avg["epoch"] == epoch)
    )
    vals = df_seed_avg.loc[mask, "score"].values
    return vals.mean(), vals.std()


def draw_fancy_bar(ax, x, height, color, hatch="", width=BAR_WIDTH):
    """Draw a bar using FancyBboxPatch with subtle top rounding.

    mutation_aspect=0.05 squashes the rounding vertically so the bottom
    stays flat while the top gets a gentle curve.
    """
    x0 = x - width / 2
    rect = mpatches.FancyBboxPatch(
        (x0, 0), width, height,
        boxstyle=mpatches.BoxStyle("Round", pad=0.02, rounding_size=0.1),
        facecolor=color,
        edgecolor="black",
        linewidth=0.8,
        hatch=hatch,
        mutation_aspect=0.05,
        zorder=3,
    )
    ax.add_patch(rect)


def draw_break_marks(ax, x):
    """Draw two diagonal slashes at the bottom spine to indicate axis break.

    Uses a blended transform: x in data coords, y in axes fraction.
    The y-center is shifted to -0.022 to align with the actual spine
    position after despine(offset=5).
    """
    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    spine_y = -0.022   # axes-fraction offset matching despine(offset=5)
    dy = 0.028
    dx = 0.20
    gap = 0.13
    # White background patch to "erase" the spine behind the break
    ax.plot([x - gap - dx - 0.1, x + gap + dx + 0.1], [spine_y, spine_y],
            color="white", linewidth=5, transform=trans, clip_on=False, zorder=9)
    for offset in [-gap, gap]:
        ax.plot(
            [x + offset - dx, x + offset + dx],
            [spine_y - dy, spine_y + dy],
            color="black", linewidth=1.8, clip_on=False, zorder=10,
            transform=trans,
        )


# ── Figure ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=False)

for i, region in enumerate(regions):
    for j, metric in enumerate(metrics):
        ax = axes[i, j]

        # Gather stats for all conditions
        m_un, s_un = get_bar_stats(metric, region, 1000, "imagenet1k", 0)
        coarse_stats = [get_bar_stats(metric, region, cfg, PCA_FOLDER, 20)
                        for cfg in COARSE_CFGS]
        m_bl, s_bl = get_bar_stats(metric, region, 1000, "imagenet1k", 20)

        all_means = np.array([m_un] + [s[0] for s in coarse_stats] + [m_bl])
        all_sds = np.array([s_un] + [s[1] for s in coarse_stats] + [s_bl])

        # ── Y-axis range (zoomed in) ────────────────────────────────────
        all_lo = min(all_means - all_sds)
        all_hi = max(all_means + all_sds)
        data_range = all_hi - all_lo
        y_bottom = max(0, all_lo - 0.20 * data_range)
        y_top = all_hi + 0.35 * data_range

        # ── Draw FancyBboxPatch bars ────────────────────────────────────
        for k_bar in range(len(ALL_X)):
            draw_fancy_bar(ax, ALL_X[k_bar], all_means[k_bar],
                           ALL_COLORS[k_bar], ALL_HATCHES[k_bar])

        # ── Error bars ──────────────────────────────────────────────────
        for k_bar in range(len(ALL_X)):
            if all_sds[k_bar] > 0:
                ax.errorbar(
                    ALL_X[k_bar], all_means[k_bar], yerr=all_sds[k_bar],
                    fmt="none", ecolor="black", elinewidth=1.0,
                    capsize=4, capthick=1.0, zorder=5,
                )

        # ── Axis break marks ────────────────────────────────────────────
        draw_break_marks(ax, BREAK_X)

        # ── Significance stars ──────────────────────────────────────────
        baseline_subj = df_subj_avg[
            (df_subj_avg["compare_method"] == metric)
            & (df_subj_avg["region"] == region)
            & (df_subj_avg["cfg_id"] == 1000)
            & (df_subj_avg["pca_labels_folder"] == "imagenet1k")
            & (df_subj_avg["epoch"] == 20)
        ].set_index("subject_idx")["score"]

        for k, cfg in enumerate(COARSE_CFGS):
            bar_idx = k + 1

            # Only annotate bars whose mean exceeds the baseline
            if all_means[bar_idx] <= all_means[-1]:
                continue

            coarse_subj = df_subj_avg[
                (df_subj_avg["compare_method"] == metric)
                & (df_subj_avg["region"] == region)
                & (df_subj_avg["cfg_id"] == cfg)
                & (df_subj_avg["pca_labels_folder"] == PCA_FOLDER)
                & (df_subj_avg["epoch"] == 20)
            ].set_index("subject_idx")["score"]

            subjects = sorted(set(coarse_subj.index) & set(baseline_subj.index))
            _, p_val = stats.ttest_rel(
                coarse_subj.loc[subjects].values,
                baseline_subj.loc[subjects].values,
            )

            if p_val < 0.001:
                sig = "***"
            elif p_val < 0.01:
                sig = "**"
            elif p_val < 0.05:
                sig = "*"
            else:
                sig = "n.s."

            y_star = all_means[bar_idx] + all_sds[bar_idx]
            ax.text(ALL_X[bar_idx], y_star + 0.02 * data_range, sig,
                    ha="center", va="bottom",
                    fontsize=11 if sig != "n.s." else 8,
                    fontweight="bold" if sig != "n.s." else "normal",
                    fontstyle="normal" if sig != "n.s." else "italic",
                    color="black")

        # ── Axis formatting ─────────────────────────────────────────────
        ax.set_xticks(ALL_X)
        ax.set_xticklabels(ALL_LABELS, fontsize=10, rotation=0, ha="center")
        ax.tick_params(axis="x", direction="out", bottom=False, length=4,
                       color="black", width=1.5)

        # Y-axis: major + minor ticks
        ax.tick_params(axis="y", which="major", direction="out", left=True,
                       right=False, labelsize=12, length=5, color="black", width=1.5)
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis="y", which="minor", direction="out", left=True,
                       right=False, length=3, color="black", width=1.0)

        ax.set_xlim(-0.6, X_BASELINE + 0.7)
        ax.set_ylim(y_bottom, y_top)

        if i == 1:
            ax.set_xlabel("Number of Classes", fontsize=14)
        if j == 0:
            ax.set_ylabel(f"{region_labels[region]}\n{metric_labels[metric]}",
                          fontsize=13, fontweight="normal")
        else:
            ax.set_ylabel(metric_labels[metric], fontsize=13, fontweight="normal")

        if i == 0:
            ax.set_title(metric_labels[metric], fontsize=16, fontweight="bold", pad=10)

# ── Despine & layout ─────────────────────────────────────────────────────────
for ax_row in axes:
    for ax in ax_row:
        sns.despine(ax=ax, right=True, top=True, offset=5)
        ax.spines["bottom"].set_linewidth(1.5)
        ax.spines["left"].set_linewidth(1.5)

fig.suptitle(
    f"Brain Alignment Across Label Granularity\n({DISPLAY_NAME}-PCA Labels, NSD RSA)",
    fontsize=18,
    fontweight="bold",
    y=1.02,
)

plt.tight_layout(pad=1.0)
OUT = f"plotters/figures/coarseness_barplot_{DISPLAY_NAME.lower()}.png"
fig.savefig(OUT, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
print(f"\nSaved → {OUT}")
plt.close()
