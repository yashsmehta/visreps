"""
Plot data efficiency results: separate figures for neural (NSD + TVSD) and
THINGS alignment across dataset sizes (mini-10, mini-100, full 1.2M).

Epoch selection per dataset:
  - imagenet-mini-10:  epoch 100
  - imagenet-mini-100: epoch 50
  - imagenet-full:     from results.db (best layer per seed)

Usage (from project root):
    python experiments/coarse_grain_benefits/data_efficiency/plot.py
"""

import os
import sys
import json
import sqlite3

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "data_efficiency_results.csv")
DB_PATH = os.path.join(PROJECT_ROOT, "results.db")

# ── Style ─────────────────────────────────────────────────────────────────
sns.set_theme(style="ticks", context="paper", font_scale=1.1)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
})

# Green shades for coarse, orange for fine-grained
COLORS = {
    8:    "#a1d99b",   # light green
    16:   "#41ab5d",   # medium green
    32:   "#006d2c",   # dark green
    64:   "#00441b",   # darkest green
    1000: "#e6550d",   # vivid orange
}
MARKERS = {8: "o", 16: "s", 32: "D", 64: "^", 1000: "X"}

CONDITIONS = [8, 16, 32, 64, 1000]
DATASETS = ["imagenet-mini-10", "imagenet-mini-100", "imagenet-full"]
DATASET_LABELS = {
    "imagenet-mini-10": "10K",
    "imagenet-mini-100": "100K",
    "imagenet-full": "1.2M",
}

# Epoch to use for each mini dataset
EPOCH_FOR_DATASET = {
    "imagenet-mini-10": 100,
    "imagenet-mini-100": 50,
}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_csv_data():
    """Load data-efficiency CSV, selecting the prescribed epoch per dataset."""
    df = pd.read_csv(CSV_PATH)
    rows = []

    for bench in ["things", "nsd", "tvsd"]:
        bdf = df[df["benchmark"] == bench]
        if bdf.empty:
            continue

        for ds in ["imagenet-mini-10", "imagenet-mini-100"]:
            epoch = EPOCH_FOR_DATASET[ds]
            dsdf = bdf[(bdf["dataset"] == ds) & (bdf["epoch"] == epoch)]

            for cond in CONDITIONS:
                cdf = dsdf[dsdf["condition"] == cond]
                if cdf.empty:
                    continue

                if bench in ("nsd", "tvsd"):
                    # Average across subjects
                    vals = {
                        "score": cdf["score"].mean(),
                        "ci_low": cdf["ci_low"].mean(),
                        "ci_high": cdf["ci_high"].mean(),
                    }
                else:
                    # THINGS: single row
                    vals = {
                        "score": cdf["score"].iloc[0],
                        "ci_low": cdf["ci_low"].iloc[0],
                        "ci_high": cdf["ci_high"].iloc[0],
                    }

                rows.append({
                    "dataset": ds, "condition": cond, "benchmark": bench, **vals,
                })

    return pd.DataFrame(rows)


def load_full_imagenet():
    """Load full ImageNet (1.2M) results from results.db."""
    conn = sqlite3.connect(DB_PATH)
    rows = []

    for cond in CONDITIONS:
        if cond == 1000:
            where = "cfg_id=1000 AND model_name='CustomCNN' AND reconstruct_from_pcs=0 AND epoch=20"
        else:
            where = f"cfg_id={cond} AND model_name='CustomCNN' AND pca_labels_folder='pca_labels_clip' AND reconstruct_from_pcs=0 AND epoch=20"

        # ── THINGS (seed 1 only, matching low-data CSV) ─────────────────
        things = pd.read_sql(f"""
            SELECT score, ci_low, ci_high
            FROM results
            WHERE neural_dataset='things-behavior'
              AND compare_method='spearman' AND seed=1 AND {where}
            ORDER BY score DESC
            LIMIT 1
        """, conn)
        if not things.empty:
            r = things.iloc[0]
            rows.append({
                "dataset": "imagenet-full", "condition": cond, "benchmark": "things",
                "score": r["score"],
                "ci_low": r["ci_low"] if pd.notna(r["ci_low"]) else r["score"],
                "ci_high": r["ci_high"] if pd.notna(r["ci_high"]) else r["score"],
            })

        # ── NSD ventral stream (seed 1 only) ──────────────────────────────
        nsd_best = pd.read_sql(f"""
            SELECT run_id, subject_idx, score,
                   ROW_NUMBER() OVER (PARTITION BY subject_idx ORDER BY score DESC) as rn
            FROM results
            WHERE neural_dataset='nsd'
              AND region='ventral visual stream'
              AND compare_method='spearman' AND seed=1 AND {where}
        """, conn)
        if not nsd_best.empty:
            nsd_best = nsd_best[nsd_best["rn"] == 1].drop(columns=["rn"])
            mean_score = nsd_best["score"].mean()
            ci_low, ci_high = _bootstrap_ci(conn, nsd_best, mean_score)
            rows.append({
                "dataset": "imagenet-full", "condition": cond, "benchmark": "nsd",
                "score": mean_score, "ci_low": ci_low, "ci_high": ci_high,
            })

        # ── TVSD IT (seed 1 only) ──────────────────────────────────────
        tvsd_best = pd.read_sql(f"""
            SELECT run_id, subject_idx, score,
                   ROW_NUMBER() OVER (PARTITION BY subject_idx ORDER BY score DESC) as rn
            FROM results
            WHERE neural_dataset='tvsd'
              AND region='IT'
              AND compare_method='spearman' AND seed=1 AND {where}
        """, conn)
        if not tvsd_best.empty:
            tvsd_best = tvsd_best[tvsd_best["rn"] == 1].drop(columns=["rn"])
            mean_score = tvsd_best["score"].mean()
            ci_low, ci_high = _bootstrap_ci(conn, tvsd_best, mean_score)
            rows.append({
                "dataset": "imagenet-full", "condition": cond, "benchmark": "tvsd",
                "score": mean_score, "ci_low": ci_low, "ci_high": ci_high,
            })

    conn.close()
    return pd.DataFrame(rows)


def _bootstrap_ci(conn, best_df, mean_score):
    """Compute 95% CI from bootstrap distributions (single seed), falling back to SEM across subjects."""
    run_ids = best_df["run_id"].unique().tolist()
    placeholders = ",".join(f"'{r}'" for r in run_ids)
    boot_dists = pd.read_sql(f"""
        SELECT bd.run_id, bd.scores
        FROM bootstrap_distributions bd
        WHERE bd.run_id IN ({placeholders})
          AND bd.compare_method='spearman'
    """, conn)

    if not boot_dists.empty:
        n_boot = 1000
        arrays = [np.array(json.loads(s)) for s in boot_dists["scores"].values]
        arrays = [a for a in arrays if len(a) == n_boot]
        if len(arrays) >= 2:
            # Average bootstrap distributions across subjects
            all_boots = np.vstack(arrays).mean(axis=0)
            return tuple(np.percentile(all_boots, [2.5, 97.5]))

    # Fallback: SEM across subjects
    sem = best_df["score"].std() / np.sqrt(len(best_df))
    return mean_score - 1.96 * sem, mean_score + 1.96 * sem


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_panel(ax, data, ylabel, title, panel_label=None):
    """Line plot for one benchmark panel."""
    x_positions = np.arange(len(DATASETS))
    x_map = {ds: i for i, ds in enumerate(DATASETS)}

    for cond in CONDITIONS:
        cdf = data[data["condition"] == cond]
        if cdf.empty:
            continue

        xs, ys, errs_lo, errs_hi = [], [], [], []
        for ds in DATASETS:
            row = cdf[cdf["dataset"] == ds]
            if row.empty:
                continue
            r = row.iloc[0]
            xs.append(x_map[ds])
            ys.append(r["score"])
            errs_lo.append(max(0, r["score"] - r["ci_low"]))
            errs_hi.append(max(0, r["ci_high"] - r["score"]))

        label = f"{cond}-class"
        ax.errorbar(xs, ys, yerr=[errs_lo, errs_hi],
                    fmt="none", ecolor=COLORS[cond], capsize=3,
                    linewidth=1.0, capthick=0.8, alpha=0.65, zorder=2)
        ax.plot(xs, ys, marker=MARKERS[cond], color=COLORS[cond],
                markersize=7, linewidth=2.0, markeredgecolor="white",
                markeredgewidth=0.7, label=label, zorder=3)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([DATASET_LABELS[ds] for ds in DATASETS], fontsize=9)
    ax.set_xlabel("Training images", fontsize=10, labelpad=6)
    ax.set_ylabel(ylabel, fontsize=10, labelpad=6)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
    ax.yaxis.grid(True, which="major", color="#ECECEC", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.margins(x=0.08)
    ax.tick_params(axis="both", which="major", direction="out", length=4, width=0.8)
    from matplotlib.ticker import AutoMinorLocator, FuncFormatter
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="minor", direction="out", length=2.5, width=0.5)
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:.2f}".rstrip("0").rstrip(".")))
    sns.despine(ax=ax, right=True, top=True, offset=5)

    if panel_label:
        ax.text(-0.14, 1.10, panel_label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="left")


def add_legend(fig, data, ncol=None):
    """Add shared legend at top of figure."""
    present = sorted(data["condition"].unique())
    handles = [
        Line2D([], [], marker=MARKERS[c], color=COLORS[c], markersize=7,
               linewidth=2.0, markeredgecolor="white", markeredgewidth=0.8,
               label=f"{c}-class")
        for c in CONDITIONS if c in present
    ]
    fig.legend(handles=handles, loc="upper center", ncol=ncol or len(handles),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.03),
               columnspacing=1.8, handletextpad=0.6)


def main():
    csv_data = load_csv_data()
    full_data = load_full_imagenet()
    data = pd.concat([csv_data, full_data], ignore_index=True)

    # ── Figure 1: Neural data (NSD ventral + TVSD IT) ────────────────────
    neural_data = data[data["benchmark"].isin(["nsd", "tvsd"])]
    fig1, axes1 = plt.subplots(1, 2, figsize=(7.5, 3.3))

    plot_panel(axes1[0], neural_data[neural_data["benchmark"] == "nsd"],
               ylabel=r"Neural alignment (Spearman $\rho$)",
               title="NSD (Ventral Stream)", panel_label="A")
    plot_panel(axes1[1], neural_data[neural_data["benchmark"] == "tvsd"],
               ylabel=r"Neural alignment (Spearman $\rho$)",
               title="TVSD (IT)", panel_label="B")

    add_legend(fig1, neural_data)
    plt.subplots_adjust(wspace=0.35)
    fig1.tight_layout(rect=[0, 0, 1, 0.90])
    out1 = os.path.join(SCRIPT_DIR, "data_efficiency_neural.png")
    fig1.savefig(out1, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved to {out1}")
    plt.close(fig1)

    # ── Figure 2: THINGS behavioral ──────────────────────────────────────
    things_data = data[data["benchmark"] == "things"]
    fig2, ax2 = plt.subplots(1, 1, figsize=(4.0, 3.3))

    plot_panel(ax2, things_data,
               ylabel=r"Behavioral alignment (Spearman $\rho$)",
               title="THINGS (Behavioral)")

    add_legend(fig2, things_data)
    fig2.tight_layout(rect=[0, 0, 1, 0.90])
    out2 = os.path.join(SCRIPT_DIR, "data_efficiency_things.png")
    fig2.savefig(out2, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved to {out2}")
    plt.close(fig2)


if __name__ == "__main__":
    main()
