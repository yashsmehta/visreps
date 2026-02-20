"""Paired comparison plots: best coarse-grained model vs baseline.

Two modes:
  MODE = "coarse_vs_fine" — SQLite, 2x2 grid (region x metric), auto-selects best cfg_id
  MODE = "architectures"  — CSV, 1x2 grid, fixed method comparison (e.g. hierarchical vs global)

Usage:
    source .venv/bin/activate && python plotters/paired_comparison.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats

# ============================================================================
# CONFIGURATION — change MODE to switch
# ============================================================================
MODE = "coarse_vs_fine"  # "coarse_vs_fine" or "architectures"

# Style
sns.set_theme(style="ticks", context="paper", font_scale=1.2)
PALETTE = {"group_a": "#4878D0", "group_b": "#D65F5F"}
SUBJECT_LINE_COLOR = "#888888"
SUBJECT_DOT_SIZE = 48
BOX_WIDTH = 0.5

# --- coarse_vs_fine settings ---
DB_PATH = "results.db"
PCA_FOLDER = "pca_labels_alexnet"
FOLDER_DISPLAY_NAME = "AlexNet"

# --- architectures settings ---
CSV_PATH = "logs/vit_global_v_hierarchical.csv"
COMPARE_COLUMN = "checkpoint_dir"
METHOD_A_VALUE = "model_checkpoints/alexnet_hierarchical_vit"
METHOD_B_VALUE = "model_checkpoints/alexnet_global_vit"
METHOD_A_LABEL = "Hierarchical"
METHOD_B_LABEL = "Global"

SUBPLOT_CONFIGS = [
    ("early visual stream", "conv4", "Early Visual Stream (Conv4)"),
    ("ventral visual stream", "fc2", "Ventral Visual Stream (FC2)"),
]


def draw_paired_panel(ax, group_a_vals, group_b_vals, label_a, label_b,
                      color_a, color_b, show_pvalue=True):
    """Draw a single paired-comparison panel: box plots + jittered dots + connecting lines + t-test bracket."""
    positions = [0, 1]
    bp = ax.boxplot(
        [group_a_vals, group_b_vals],
        positions=positions,
        widths=BOX_WIDTH,
        patch_artist=True,
        showfliers=False,
        zorder=2,
    )
    colors = [color_a, color_b]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)
    for element in ["whiskers", "caps"]:
        for line, color in zip(bp[element], np.repeat(colors, 2)):
            line.set_color(color)
            line.set_linewidth(1.4)
    for med, color in zip(bp["medians"], colors):
        med.set_color(color)
        med.set_linewidth(2.0)

    # Paired subject lines + dots
    jitter = 0.04
    rng = np.random.default_rng(42)
    n = len(group_a_vals)
    x_jitter = rng.uniform(-jitter, jitter, size=n)

    for k in range(n):
        ax.plot([0 + x_jitter[k], 1 + x_jitter[k]],
                [group_a_vals[k], group_b_vals[k]],
                color=SUBJECT_LINE_COLOR, linewidth=0.8, alpha=0.6, zorder=3)

    ax.scatter(np.zeros(n) + x_jitter, group_a_vals, s=SUBJECT_DOT_SIZE,
               color=color_a, edgecolors="white", linewidths=0.6, zorder=4)
    ax.scatter(np.ones(n) + x_jitter, group_b_vals, s=SUBJECT_DOT_SIZE,
               color=color_b, edgecolors="white", linewidths=0.6, zorder=4)

    # Paired t-test bracket
    t_stat, p_val = stats.ttest_rel(group_a_vals, group_b_vals)
    if p_val < 0.001:
        sig_text = "***"
    elif p_val < 0.01:
        sig_text = "**"
    elif p_val < 0.05:
        sig_text = "*"
    else:
        sig_text = "n.s."

    y_max = max(group_a_vals.max(), group_b_vals.max())
    y_range = y_max - min(group_a_vals.min(), group_b_vals.min())
    bracket_y = y_max + 0.06 * y_range
    bracket_h = 0.02 * y_range
    ax.plot([0, 0, 1, 1],
            [bracket_y, bracket_y + bracket_h, bracket_y + bracket_h, bracket_y],
            color="black", linewidth=1.2, zorder=5)

    annotation = f"{sig_text}\np = {p_val:.3f}" if show_pvalue else sig_text
    ax.text(0.5, bracket_y + bracket_h + 0.01 * y_range, annotation,
            ha="center", va="bottom", fontsize=10, color="black", linespacing=1.1)

    ax.set_xticks(positions)

    return t_stat, p_val


def run_coarse_vs_fine():
    """2x2 paired plot: coarse vs 1K baseline from SQLite, rows=region, cols=metric."""
    import sqlite3

    QUERY = f"""
    SELECT compare_method, score, seed, region, subject_idx, cfg_id, pca_labels_folder
    FROM results
    WHERE (pca_labels_folder = '{PCA_FOLDER}' OR pca_labels_folder = 'imagenet1k')
      AND compare_method IN ('spearman', 'kendall')
    """
    print(f"SQL query:\n{QUERY}")

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(QUERY, conn)
    conn.close()

    # Average over seeds
    df_avg = (
        df.groupby(["compare_method", "region", "subject_idx", "cfg_id", "pca_labels_folder"])["score"]
        .mean()
        .reset_index()
    )

    regions = ["early visual stream", "ventral visual stream"]
    metrics = ["spearman", "kendall"]
    region_labels = {"early visual stream": "Early Visual Stream",
                     "ventral visual stream": "Ventral Visual Stream"}
    metric_labels = {"spearman": "Spearman \u03c1", "kendall": "Kendall \u03c4"}

    # Select best coarse cfg_id per (region, metric)
    best_configs = {}
    for region in regions:
        for metric in metrics:
            coarse = df_avg[
                (df_avg["region"] == region)
                & (df_avg["compare_method"] == metric)
                & (df_avg["pca_labels_folder"] == PCA_FOLDER)
            ]
            mean_by_cfg = coarse.groupby("cfg_id")["score"].mean()
            best_cfg = int(mean_by_cfg.idxmax())
            best_configs[(region, metric)] = best_cfg
            print(f"Best coarse model for {region} / {metric}: "
                  f"{FOLDER_DISPLAY_NAME} cfg_id={best_cfg} (mean={mean_by_cfg.max():.4f})")

    fig, axes = plt.subplots(2, 2, figsize=(7, 7), sharey=False)

    for i, region in enumerate(regions):
        for j, metric in enumerate(metrics):
            ax = axes[i, j]
            best_cfg = best_configs[(region, metric)]

            coarse = df_avg[
                (df_avg["region"] == region)
                & (df_avg["compare_method"] == metric)
                & (df_avg["cfg_id"] == best_cfg)
                & (df_avg["pca_labels_folder"] == PCA_FOLDER)
            ].set_index("subject_idx")["score"]

            baseline = df_avg[
                (df_avg["region"] == region)
                & (df_avg["compare_method"] == metric)
                & (df_avg["cfg_id"] == 1000)
                & (df_avg["pca_labels_folder"] == "imagenet1k")
            ].set_index("subject_idx")["score"]

            subjects = sorted(set(coarse.index) & set(baseline.index))
            coarse_vals = coarse.loc[subjects].values
            baseline_vals = baseline.loc[subjects].values

            draw_paired_panel(ax, coarse_vals, baseline_vals,
                              f"{FOLDER_DISPLAY_NAME} ({best_cfg})", "ImageNet (1000)",
                              PALETTE["group_a"], PALETTE["group_b"])

            ax.set_xticklabels(
                [f"{FOLDER_DISPLAY_NAME} ({best_cfg})", "ImageNet (1000)"],
                fontsize=13,
            )
            ax.tick_params(axis="y", labelsize=13)
            ax.yaxis.set_major_locator(ticker.MaxNLocator(5))

            if j == 0:
                ax.set_ylabel(f"{region_labels[region]}\n{metric_labels[metric]}",
                              fontsize=14, fontweight="normal")
            else:
                ax.set_ylabel(metric_labels[metric], fontsize=14, fontweight="normal")

            if i == 0:
                ax.set_title(metric_labels[metric], fontsize=16, fontweight="bold", pad=12)

            y_lo, y_hi = ax.get_ylim()
            ax.set_ylim(y_lo, y_hi + 0.12 * (y_hi - y_lo))

    for ax_row in axes:
        for ax in ax_row:
            sns.despine(ax=ax, right=True, top=True, offset=8)
            for spine in ax.spines.values():
                spine.set_linewidth(1.8)

    fig.suptitle(
        f"Coarse vs. Fine-Grained Supervision\n({FOLDER_DISPLAY_NAME}-PCA Labels, NSD RSA)",
        fontsize=18, fontweight="bold", y=1.02,
    )

    plt.tight_layout()
    OUT = f"plotters/figures/paired_comparison_{FOLDER_DISPLAY_NAME.lower()}_vs_fine.png"
    fig.savefig(OUT, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"\nSaved -> {OUT}")
    plt.close()


def run_architectures():
    """1x2 paired plot: method A vs method B from CSV, cols=region."""
    df = pd.read_csv(CSV_PATH)
    df['layer'] = df['layer'].str.lower()
    df['region'] = df['region'].str.lower()

    fig, axes = plt.subplots(1, 2, figsize=(10, 6))
    colors = {METHOD_A_LABEL: '#1f77b4', METHOD_B_LABEL: '#ff7f0e'}

    for ax_idx, (region, layer, title) in enumerate(SUBPLOT_CONFIGS):
        ax = axes[ax_idx]

        df_region = df[(df['region'] == region) & (df['layer'] == layer)]
        df_a = df_region[df_region[COMPARE_COLUMN] == METHOD_A_VALUE].sort_values('subject_idx').reset_index(drop=True)
        df_b = df_region[df_region[COMPARE_COLUMN] == METHOD_B_VALUE].sort_values('subject_idx').reset_index(drop=True)

        if len(df_a) == 0 or len(df_b) == 0:
            print(f"Warning: Missing data for {region}, {layer}")
            continue

        scores_a = df_a['score'].values
        scores_b = df_b['score'].values
        subject_ids = df_a['subject_idx'].values

        print(f"\n{title}:")
        print(f"  {METHOD_A_LABEL}: mean={np.mean(scores_a):.4f}, std={np.std(scores_a):.4f}")
        print(f"  {METHOD_B_LABEL}: mean={np.mean(scores_b):.4f}, std={np.std(scores_b):.4f}")

        t_stat, p_val = draw_paired_panel(
            ax, scores_a, scores_b,
            METHOD_A_LABEL, METHOD_B_LABEL,
            colors[METHOD_A_LABEL], colors[METHOD_B_LABEL],
            show_pvalue=False,
        )
        print(f"  Paired t-test: t={t_stat:.4f}, p={p_val:.4f}")

        ax.set_xticklabels([METHOD_A_LABEL, METHOD_B_LABEL], fontsize=12, fontweight='bold')
        ax.set_ylabel('Brain Similarity (RSA)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)

        all_scores = np.concatenate([scores_a, scores_b])
        ax.set_ylim(0, max(all_scores) + 0.05)

    plt.tight_layout(pad=2.0)
    OUTPUT_PATH = "plotters/figures/paired_comparison_architectures.png"
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"\nPlot saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    if MODE == "coarse_vs_fine":
        run_coarse_vs_fine()
    elif MODE == "architectures":
        run_architectures()
