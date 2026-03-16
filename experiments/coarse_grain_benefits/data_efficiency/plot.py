"""
Plot data efficiency results: line plots for THINGS and NSD alignment
across dataset sizes (5K → 1.2M), colored by supervision granularity.

- Data-efficiency results (5K/10K/50K) from CSV
- Full ImageNet (1.2M) results from results.db

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
OUT_PATH = os.path.join(SCRIPT_DIR, "data_efficiency.png")

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
    1000: "#e6550d",   # vivid orange
}
MARKERS = {8: "o", 16: "s", 32: "D", 1000: "X"}

CONDITIONS = [8, 16, 32, 1000]
DATASETS = ["imagenet-mini-5", "imagenet-mini-10", "imagenet-mini-50", "imagenet-full"]
DATASET_LABELS = {
    "imagenet-mini-5": "5K", "imagenet-mini-10": "10K",
    "imagenet-mini-50": "50K", "imagenet-full": "1.2M",
}
BENCHMARKS = {
    "things": {
        "title": "THINGS (Behavioral)",
        "ylabel": r"Behavioral alignment (Spearman $\rho$)",
    },
    "nsd": {
        "title": "NSD (Ventral Stream)",
        "ylabel": r"Neural alignment (Spearman $\rho$)",
    },
}


# ── Data loading ─────────────────────────────────────────────────────────────

def load_csv_data():
    """Load data-efficiency CSV and aggregate to one row per (dataset, condition, benchmark)."""
    df = pd.read_csv(CSV_PATH)
    rows = []

    for bench in ["things", "nsd"]:
        bdf = df[df["benchmark"] == bench]
        if bdf.empty:
            continue

        for ds in ["imagenet-mini-5", "imagenet-mini-10", "imagenet-mini-50"]:
            dsdf = bdf[bdf["dataset"] == ds]
            for cond in CONDITIONS:
                cdf = dsdf[dsdf["condition"] == cond]
                if cdf.empty:
                    continue

                if bench == "nsd":
                    # Average across subjects per epoch, then pick best epoch
                    epoch_scores = {}
                    for epoch, edf in cdf.groupby("epoch"):
                        epoch_scores[epoch] = {
                            "score": edf["score"].mean(),
                            "ci_low": edf["ci_low"].mean(),
                            "ci_high": edf["ci_high"].mean(),
                        }
                    best_epoch = max(epoch_scores, key=lambda e: epoch_scores[e]["score"])
                    vals = epoch_scores[best_epoch]
                else:
                    # THINGS: pick best epoch directly
                    best_row = cdf.loc[cdf["score"].idxmax()]
                    vals = {
                        "score": best_row["score"],
                        "ci_low": best_row["ci_low"],
                        "ci_high": best_row["ci_high"],
                    }

                rows.append({
                    "dataset": ds, "condition": cond, "benchmark": bench, **vals,
                })

    return pd.DataFrame(rows)


def load_full_imagenet():
    """Load full ImageNet (1.2M) results from results.db with proper bootstrap CIs."""
    conn = sqlite3.connect(DB_PATH)
    rows = []

    for cond in CONDITIONS:
        if cond == 1000:
            where = "cfg_id=1000 AND reconstruct_from_pcs=0"
        else:
            where = f"cfg_id={cond} AND pca_labels_folder='pca_labels_clip' AND reconstruct_from_pcs=0"

        # ── THINGS ───────────────────────────────────────────────────────
        # Best layer per seed, then average across seeds
        things = pd.read_sql(f"""
            SELECT seed, score, ci_low, ci_high
            FROM results
            WHERE neural_dataset='things-behavior'
              AND compare_method='spearman' AND {where}
            ORDER BY seed, score DESC
        """, conn)
        if not things.empty:
            # Keep best layer per seed
            best = things.groupby("seed").first().reset_index()
            rows.append({
                "dataset": "imagenet-full", "condition": cond, "benchmark": "things",
                "score": best["score"].mean(),
                "ci_low": best["ci_low"].mean(),
                "ci_high": best["ci_high"].mean(),
            })

        # ── NSD ventral stream ───────────────────────────────────────────
        # Best run_id per (seed, subject), then use bootstrap distributions
        nsd_best = pd.read_sql(f"""
            SELECT run_id, seed, subject_idx, score,
                   ROW_NUMBER() OVER (PARTITION BY seed, subject_idx ORDER BY score DESC) as rn
            FROM results
            WHERE neural_dataset='nsd'
              AND region='ventral visual stream'
              AND compare_method='spearman' AND {where}
        """, conn)
        if nsd_best.empty:
            continue
        nsd_best = nsd_best[nsd_best["rn"] == 1].drop(columns=["rn"])
        mean_score = nsd_best["score"].mean()

        # Get bootstrap distributions for best runs and average across subjects
        run_ids = nsd_best["run_id"].unique().tolist()
        placeholders = ",".join(f"'{r}'" for r in run_ids)
        boot_dists = pd.read_sql(f"""
            SELECT bd.run_id, bd.scores
            FROM bootstrap_distributions bd
            WHERE bd.run_id IN ({placeholders})
              AND bd.compare_method='spearman'
        """, conn)
        # Merge seed/subject info
        boot_dists = boot_dists.merge(
            nsd_best[["run_id", "seed", "subject_idx"]], on="run_id"
        )

        if not boot_dists.empty:
            # For each seed: average 1000 bootstrap values across 8 subjects
            n_boot = 1000
            seed_boots = []
            for seed, sdf in boot_dists.groupby("seed"):
                arrays = [np.array(json.loads(s)) for s in sdf["scores"].values]
                arrays = [a for a in arrays if len(a) == n_boot]
                if len(arrays) < 2:
                    continue
                seed_boots.append(np.vstack(arrays).mean(axis=0))
            all_boots = np.vstack(seed_boots).mean(axis=0) if seed_boots else None
            if all_boots is not None:
                ci_low, ci_high = np.percentile(all_boots, [2.5, 97.5])
            else:
                sem = nsd_best["score"].std() / np.sqrt(len(nsd_best))
                ci_low = mean_score - 1.96 * sem
                ci_high = mean_score + 1.96 * sem
        else:
            sem = nsd_best["score"].std() / np.sqrt(len(nsd_best))
            ci_low = mean_score - 1.96 * sem
            ci_high = mean_score + 1.96 * sem

        rows.append({
            "dataset": "imagenet-full", "condition": cond, "benchmark": "nsd",
            "score": mean_score, "ci_low": ci_low, "ci_high": ci_high,
        })

    conn.close()
    return pd.DataFrame(rows)


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_panel(ax, data, benchmark, panel_label=None):
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
            errs_lo.append(r["score"] - r["ci_low"])
            errs_hi.append(r["ci_high"] - r["score"])

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
    ax.set_ylabel(BENCHMARKS[benchmark]["ylabel"], fontsize=10, labelpad=6)
    ax.set_title(BENCHMARKS[benchmark]["title"], fontsize=11, fontweight="bold", pad=10)
    ax.yaxis.grid(True, which="major", color="#ECECEC", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.margins(x=0.08)
    ax.tick_params(axis="both", which="major", direction="out", length=4, width=0.8)
    from matplotlib.ticker import AutoMinorLocator, FuncFormatter
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="minor", direction="out", length=2.5, width=0.5)
    # Clean y-axis: drop trailing zeros (e.g. 0.10 → 0.1)
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:.2f}".rstrip("0").rstrip(".")))
    sns.despine(ax=ax, right=True, top=True, offset=5)

    # Panel label
    if panel_label:
        ax.text(-0.14, 1.10, panel_label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="left")


def main():
    csv_data = load_csv_data()
    full_data = load_full_imagenet()
    data = pd.concat([csv_data, full_data], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.3))

    panel_labels = ["A", "B"]
    for i, bench in enumerate(["things", "nsd"]):
        plot_panel(axes[i], data[data["benchmark"] == bench], bench,
                   panel_label=panel_labels[i])

    # Shared legend at top
    handles = [
        Line2D([], [], marker=MARKERS[c], color=COLORS[c], markersize=7,
               linewidth=2.0, markeredgecolor="white", markeredgewidth=0.8,
               label=f"{c}-class")
        for c in CONDITIONS if c in data["condition"].unique()
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.03),
               columnspacing=1.8, handletextpad=0.6)

    plt.subplots_adjust(wspace=0.35)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(OUT_PATH, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    print(f"Saved to {OUT_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
