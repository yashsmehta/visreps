"""
Reconstruction Analysis Plotter
================================
Plots RSA score vs. pca_k for the 1000-way model with reconstruct_from_pcs=True,
overlaid with horizontal baselines:
  1. 1000-way trained AlexNet (no reconstruction) — full model performance
  2. Best coarse-grained model across all label sources (AlexNet, CLIP, etc.)

All data is read from the global results.db at the project root.

Produces one figure per neural dataset:
  - NSD:    1x2 subplots (early visual stream, ventral visual stream)
  - TVSD:   1x3 subplots (V1, V4, IT)
  - THINGS: single panel
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

DB_PATH = "results.db"
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


# ── Data queries ──────────────────────────────────────────────────────────────

def query_reconstruction_curve(neural_dataset, region=None):
    """Get per-(pca_k, seed, subject) best-layer scores for reconstruction runs."""
    conn = sqlite3.connect(DB_PATH)
    q = """
    SELECT pca_k, seed, subject_idx, layer, score
    FROM results
    WHERE reconstruct_from_pcs = 1
      AND cfg_id = 1000
      AND analysis = 'rsa'
      AND compare_method = 'spearman'
      AND neural_dataset = ?
    """
    params = [neural_dataset]
    if region is not None:
        q += " AND region = ?"
        params.append(region)
    df = pd.read_sql(q, conn, params=params)
    conn.close()

    if df.empty:
        return df

    # Best layer per (pca_k, seed, subject_idx)
    idx = df.groupby(["pca_k", "seed", "subject_idx"])["score"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def query_1000way_baseline(neural_dataset, region=None):
    """Get 1000-way trained model scores WITHOUT reconstruction.

    Returns (mean, sem) across seeds (averaging over subjects within each seed).
    """
    conn = sqlite3.connect(DB_PATH)
    q = """
    SELECT seed, subject_idx, layer, score
    FROM results
    WHERE cfg_id = 1000
      AND reconstruct_from_pcs = 0
      AND analysis = 'rsa'
      AND compare_method = 'spearman'
      AND neural_dataset = ?
    """
    params = [neural_dataset]
    if region is not None:
        q += " AND region = ?"
        params.append(region)
    df = pd.read_sql(q, conn, params=params)
    conn.close()

    if df.empty:
        return np.nan, np.nan

    # Best layer per (seed, subject_idx) — results table stores only best layer
    idx = df.groupby(["seed", "subject_idx"])["score"].idxmax()
    best = df.loc[idx]

    # Average across subjects within each seed, then mean/SEM across seeds
    seed_means = best.groupby("seed")["score"].mean()
    mean = seed_means.mean()
    sem = seed_means.std() / np.sqrt(len(seed_means)) if len(seed_means) > 1 else 0.0
    return mean, sem


def query_coarse_baseline(neural_dataset, region=None):
    """Get best coarse-grained model's grand-mean score across ALL label sources.

    Searches across all checkpoint_dirs (alexnet_pca, clip_pca, dino_pca, vit_pca).
    Returns (mean, sem) for the best (checkpoint_dir, cfg_id) combo.
    """
    conn = sqlite3.connect(DB_PATH)
    q = """
    SELECT checkpoint_dir, cfg_id, seed, subject_idx, layer, score
    FROM results
    WHERE reconstruct_from_pcs = 0
      AND pca_labels = 1
      AND cfg_id IN (2, 4, 8, 16, 32, 64)
      AND analysis = 'rsa'
      AND compare_method = 'spearman'
      AND neural_dataset = ?
    """
    params = [neural_dataset]
    if region is not None:
        q += " AND region = ?"
        params.append(region)
    df = pd.read_sql(q, conn, params=params)
    conn.close()

    if df.empty:
        return np.nan, np.nan

    # Best layer per (checkpoint_dir, cfg_id, seed, subject_idx)
    idx = df.groupby(["checkpoint_dir", "cfg_id", "seed", "subject_idx"])["score"].idxmax()
    best = df.loc[idx]

    # Grand mean per (checkpoint_dir, cfg_id): avg over subjects, then over seeds
    seed_means = best.groupby(["checkpoint_dir", "cfg_id", "seed"])["score"].mean().reset_index()
    grand = seed_means.groupby(["checkpoint_dir", "cfg_id"])["score"].agg(["mean", "std", "count"]).reset_index()
    grand["sem"] = grand["std"] / np.sqrt(grand["count"])

    # Pick the best combo
    best_idx = grand["mean"].idxmax()
    return grand.loc[best_idx, "mean"], grand.loc[best_idx, "sem"]


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_curve(df):
    """Aggregate reconstruction curve: mean +/- SEM across seeds.

    Steps:
      1. Per (pca_k, seed): average score across subjects
      2. Per pca_k: mean and SEM across seeds
    """
    if df.empty:
        return pd.DataFrame(columns=["pca_k", "mean", "sem"])

    # Average across subjects within each seed
    seed_means = df.groupby(["pca_k", "seed"])["score"].mean().reset_index()

    # Mean and SEM across seeds
    agg = seed_means.groupby("pca_k")["score"].agg(["mean", "std", "count"]).reset_index()
    agg["sem"] = agg["std"] / np.sqrt(agg["count"])
    return agg[["pca_k", "mean", "sem"]]


# ── Plotting ──────────────────────────────────────────────────────────────────

CURVE_COLOR = "#1f77b4"
TRAINED_COLOR = "#2ca02c"
COARSE_COLOR = "#d62728"


def plot_panel(ax, curve_df, trained_baseline, coarse_baseline, title, show_ylabel=True):
    """Plot one reconstruction curve panel with baselines."""
    k = curve_df["pca_k"].values
    mean = curve_df["mean"].values
    sem = curve_df["sem"].values

    # Reconstruction curve
    ax.plot(k, mean, "-o", color=CURVE_COLOR, markersize=4, linewidth=1.5,
            label="1000-way (top-$k$ PCs)", zorder=3)
    ax.fill_between(k, mean - sem, mean + sem, color=CURVE_COLOR, alpha=0.15, zorder=2)

    # 1000-way trained baseline (no reconstruction)
    trained_mean, trained_sem = trained_baseline
    if not np.isnan(trained_mean):
        ax.axhline(trained_mean, color=TRAINED_COLOR, linestyle="-", linewidth=1.5,
                   label="1000-way (all dims)", zorder=1)
        if trained_sem > 0:
            ax.axhspan(trained_mean - trained_sem, trained_mean + trained_sem,
                       color=TRAINED_COLOR, alpha=0.10, zorder=0)

    # Best coarse-grained model baseline
    coarse_mean, coarse_sem = coarse_baseline
    if not np.isnan(coarse_mean):
        ax.axhline(coarse_mean, color=COARSE_COLOR, linestyle="--", linewidth=1.5,
                   label="Best coarse model", zorder=1)
        if coarse_sem > 0:
            ax.axhspan(coarse_mean - coarse_sem, coarse_mean + coarse_sem,
                       color=COARSE_COLOR, alpha=0.10, zorder=0)

    ax.set_xlabel("Number of PCs ($k$)", fontsize=10)
    if show_ylabel:
        ax.set_ylabel("RSA Score (Spearman)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.tick_params(labelsize=9)


def plot_dataset(neural_dataset, regions, figsize):
    """Produce a multi-panel figure for one neural dataset."""
    sns.set_theme(style="ticks", context="paper", font_scale=1.1)

    n = len(regions)
    fig, axes = plt.subplots(1, n, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i, (region, region_label) in enumerate(regions):
        curve_df = query_reconstruction_curve(neural_dataset, region)
        agg = aggregate_curve(curve_df)
        trained_baseline = query_1000way_baseline(neural_dataset, region)
        coarse_baseline = query_coarse_baseline(neural_dataset, region)

        if agg.empty:
            axes[i].text(0.5, 0.5, "No data", ha="center", va="center",
                         transform=axes[i].transAxes, fontsize=12)
            axes[i].set_title(region_label, fontsize=11, fontweight="bold")
            continue

        plot_panel(axes[i], agg, trained_baseline, coarse_baseline,
                   region_label, show_ylabel=(i == 0))

    # Shared legend from first plotted axis
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
                   frameon=True, edgecolor="black", fancybox=False,
                   bbox_to_anchor=(0.5, -0.02))

    sns.despine()
    plt.tight_layout(rect=[0, 0.06, 1, 1])

    out_path = FIGURES_DIR / f"reconstruction_{neural_dataset}.png"
    plt.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # NSD: 1x2
    plot_dataset("nsd", [
        ("early visual stream", "Early Visual Stream"),
        ("ventral visual stream", "Ventral Visual Stream"),
    ], figsize=(8, 3.5))

    # TVSD: 1x3
    plot_dataset("tvsd", [
        ("V1", "V1"),
        ("V4", "V4"),
        ("IT", "IT"),
    ], figsize=(11, 3.5))

    # THINGS: single panel
    plot_dataset("things-behavior", [
        ("N/A", "THINGS Behavior"),
    ], figsize=(4.5, 3.5))


if __name__ == "__main__":
    main()
