"""Figure 6: Data Efficiency — Coarse vs Fine-Grained Training.

Layout (3 panels):
  Panel A: NSD (Early Visual Stream) — line plot across 4 data scales
  Panel B: NSD (Ventral Stream) — line plot across 4 data scales
  Panel C: THINGS (Behavioral) — line plot across 4 data scales

Usage:
    python manuscript/figures/fig6/figure6.py
"""

import os
import sys
import json
import sqlite3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from matplotlib.ticker import AutoMinorLocator, FuncFormatter
import seaborn as sns

sys.path.insert(0, "manuscript/figures")
from fig_utils import setup_style

# ── Config ────────────────────────────────────────────────────────────────
OUTPUT_DIR = "manuscript/figures/fig6"
DB_PATH = "results.db"
DATA_EFF_CSV = os.path.join("experiments", "coarse_grain_benefits",
                            "data_efficiency", "data_efficiency_results.csv")

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
    "nsd_early": {
        "title": "NSD (Early Visual Stream)",
        "ylabel": r"RSA (Spearman $\rho$)",
    },
    "nsd": {
        "title": "NSD (Ventral Stream)",
        "ylabel": r"RSA (Spearman $\rho$)",
    },
    "things": {
        "title": "THINGS (Behavioral)",
        "ylabel": r"RSA (Spearman $\rho$)",
    },
}


# ── Data loading ─────────────────────────────────────────────────────────

def load_csv_data():
    """Load data-efficiency CSV and aggregate to one row per (dataset, condition, benchmark)."""
    df = pd.read_csv(DATA_EFF_CSV)
    rows = []

    for bench in ["nsd", "nsd_early", "things"]:
        bdf = df[df["benchmark"] == bench]
        if bdf.empty:
            continue

        for ds in ["imagenet-mini-5", "imagenet-mini-10", "imagenet-mini-50"]:
            dsdf = bdf[bdf["dataset"] == ds]
            for cond in CONDITIONS:
                cdf = dsdf[dsdf["condition"] == cond]
                if cdf.empty:
                    continue

                if bench in ("nsd", "nsd_early"):
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
        things = pd.read_sql(f"""
            SELECT seed, score, ci_low, ci_high
            FROM results
            WHERE neural_dataset='things-behavior'
              AND compare_method='spearman' AND {where}
            ORDER BY seed, score DESC
        """, conn)
        if not things.empty:
            best = things.groupby("seed").first().reset_index()
            rows.append({
                "dataset": "imagenet-full", "condition": cond, "benchmark": "things",
                "score": best["score"].mean(),
                "ci_low": best["ci_low"].mean(),
                "ci_high": best["ci_high"].mean(),
            })

        # ── NSD ventral stream ───────────────────────────────────────────
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

        # Get bootstrap distributions for best runs
        run_ids = nsd_best["run_id"].unique().tolist()
        placeholders = ",".join(f"'{r}'" for r in run_ids)
        boot_dists = pd.read_sql(f"""
            SELECT bd.run_id, bd.scores
            FROM bootstrap_distributions bd
            WHERE bd.run_id IN ({placeholders})
              AND bd.compare_method='spearman'
        """, conn)
        boot_dists = boot_dists.merge(
            nsd_best[["run_id", "seed", "subject_idx"]], on="run_id"
        )

        if not boot_dists.empty:
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

        # ── NSD early visual stream ────────────────────────────────────
        nsd_early = pd.read_sql(f"""
            SELECT run_id, seed, subject_idx, score,
                   ROW_NUMBER() OVER (PARTITION BY seed, subject_idx ORDER BY score DESC) as rn
            FROM results
            WHERE neural_dataset='nsd'
              AND region='early visual stream'
              AND compare_method='spearman' AND {where}
        """, conn)
        if not nsd_early.empty:
            nsd_early = nsd_early[nsd_early["rn"] == 1].drop(columns=["rn"])
            early_mean = nsd_early["score"].mean()

            early_run_ids = nsd_early["run_id"].unique().tolist()
            early_ph = ",".join(f"'{r}'" for r in early_run_ids)
            early_boots = pd.read_sql(f"""
                SELECT bd.run_id, bd.scores
                FROM bootstrap_distributions bd
                WHERE bd.run_id IN ({early_ph})
                  AND bd.compare_method='spearman'
            """, conn)
            early_boots = early_boots.merge(
                nsd_early[["run_id", "seed", "subject_idx"]], on="run_id"
            )

            if not early_boots.empty:
                early_seed_boots = []
                for seed, sdf in early_boots.groupby("seed"):
                    arrays = [np.array(json.loads(s)) for s in sdf["scores"].values]
                    arrays = [a for a in arrays if len(a) == 1000]
                    if len(arrays) < 2:
                        continue
                    early_seed_boots.append(np.vstack(arrays).mean(axis=0))
                early_all = np.vstack(early_seed_boots).mean(axis=0) if early_seed_boots else None
                if early_all is not None:
                    e_ci_low, e_ci_high = np.percentile(early_all, [2.5, 97.5])
                else:
                    sem = nsd_early["score"].std() / np.sqrt(len(nsd_early))
                    e_ci_low = early_mean - 1.96 * sem
                    e_ci_high = early_mean + 1.96 * sem
            else:
                sem = nsd_early["score"].std() / np.sqrt(len(nsd_early))
                e_ci_low = early_mean - 1.96 * sem
                e_ci_high = early_mean + 1.96 * sem

            rows.append({
                "dataset": "imagenet-full", "condition": cond, "benchmark": "nsd_early",
                "score": early_mean, "ci_low": e_ci_low, "ci_high": e_ci_high,
            })

    conn.close()
    return pd.DataFrame(rows)


# ── Plotting ─────────────────────────────────────────────────────────────

def plot_panel(ax, data, benchmark):
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
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="minor", direction="out", length=2.5, width=0.5)
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:.2f}".rstrip("0").rstrip(".")))
    sns.despine(ax=ax, right=True, top=True, offset=5)


def main():
    setup_style()

    csv_data = load_csv_data()
    full_data = load_full_imagenet()
    data = pd.concat([csv_data, full_data], ignore_index=True)

    fig = plt.figure(figsize=(12, 3.5))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35,
                           left=0.06, right=0.97, top=0.85, bottom=0.15)

    # Panel A: NSD (Early Visual Stream)
    ax_early = fig.add_subplot(gs[0, 0])
    plot_panel(ax_early, data[data["benchmark"] == "nsd_early"], "nsd_early")

    # Panel B: NSD (Ventral Stream)
    ax_nsd = fig.add_subplot(gs[0, 1])
    plot_panel(ax_nsd, data[data["benchmark"] == "nsd"], "nsd")

    # Panel C: THINGS (Behavioral)
    ax_things = fig.add_subplot(gs[0, 2])
    plot_panel(ax_things, data[data["benchmark"] == "things"], "things")

    # Panel labels
    for ax, label, x_off in zip(
        [ax_early, ax_nsd, ax_things],
        ["A", "B", "C"],
        [-0.14, -0.14, -0.14],
    ):
        ax.text(x_off, 1.12, label, transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top", ha="left",
                family="sans-serif")

    # Shared legend at top
    handles = [
        Line2D([], [], marker=MARKERS[c], color=COLORS[c], markersize=7,
               linewidth=2.0, markeredgecolor="white", markeredgewidth=0.8,
               label=f"{c}-class")
        for c in CONDITIONS if c in data["condition"].unique()
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles),
               frameon=False, fontsize=9, bbox_to_anchor=(0.52, 1.02),
               columnspacing=1.8, handletextpad=0.6)

    out = f"{OUTPUT_DIR}/figure6.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white",
                edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()
