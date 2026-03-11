"""Shared utilities for reconstruction analysis plots."""

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DB_PATH = "results.db"
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
COMPARE_METHOD = "spearman"

# ── Style ────────────────────────────────────────────────────────────────────

FINE_COLOR = "#e6a200"        # golden amber — 1000-way reconstruction curve
TRAINED_COLOR = "#d45500"     # burnt orange-red — 1000-way baseline
COARSE_COLOR = "#2166ac"      # blue — coarse model reconstruction curve
UNTRAINED_COLOR = "#969696"   # neutral grey
# Keep old name as alias for backward compat (used by plot_case_study.py)
CURVE_COLOR = FINE_COLOR

SPINE_WIDTH = 1.0
TICK_MAJOR_LEN = 4
TICK_MINOR_LEN = 2.5

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Nimbus Sans", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans",
    "mathtext.it": "Liberation Sans:italic",
}

DATASET_LAYOUTS = {
    "nsd": {
        "regions": [("early visual stream", "Early Visual Stream"),
                    ("ventral visual stream", "Ventral Visual Stream")],
        "figsize": (7.5, 3.2),
    },
    "tvsd": {
        "regions": [("V1", "V1"), ("V4", "V4"), ("IT", "IT")],
        "figsize": (10, 3.2),
    },
    "things-behavior": {
        "regions": [("N/A", "THINGS Behavior")],
        "figsize": (4, 3.2),
    },
}


# ── Bootstrap CI ─────────────────────────────────────────────────────────────

def get_bootstrap_ci(run_ids, alpha=0.05):
    """Average bootstrap distributions element-wise across runs → (mean, lo, hi)."""
    if not run_ids:
        return np.nan, np.nan, np.nan

    conn = sqlite3.connect(DB_PATH)
    ph = ",".join("?" for _ in run_ids)
    rows = conn.execute(
        f"SELECT scores FROM bootstrap_distributions "
        f"WHERE run_id IN ({ph}) AND compare_method = ?",
        list(run_ids) + [COMPARE_METHOD],
    ).fetchall()
    conn.close()

    if not rows:
        return np.nan, np.nan, np.nan

    arrays = [np.array(json.loads(r[0])) for r in rows]
    min_len = min(len(a) for a in arrays)
    mean_dist = np.mean([a[:min_len] for a in arrays], axis=0)
    return (float(np.mean(mean_dist)),
            float(np.percentile(mean_dist, 100 * alpha / 2)),
            float(np.percentile(mean_dist, 100 * (1 - alpha / 2))))


# ── Generic query helpers ────────────────────────────────────────────────────

def _query_best_scores(where_clause, params):
    """Query results.db, pick best layer per (seed, subject_idx), return (mean, ci_lo, ci_hi)."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        f"SELECT run_id, seed, subject_idx, layer, score FROM results WHERE {where_clause}",
        conn, params=params,
    )
    conn.close()

    if df.empty:
        return np.nan, np.nan, np.nan

    best = df.loc[df.groupby(["seed", "subject_idx"])["score"].idxmax()]
    _, ci_low, ci_high = get_bootstrap_ci(best["run_id"].tolist())
    return best["score"].mean(), ci_low, ci_high


def _region_filter(region):
    """Return (sql_fragment, params) for optional region filtering."""
    if region is not None:
        return " AND region = ?", [region]
    return "", []


def query_reconstruction_curve(neural_dataset, region=None,
                               cfg_id=1000, checkpoint_dir=None):
    """Per-(pca_k, seed, subject) best-layer scores for reconstruction runs."""
    conn = sqlite3.connect(DB_PATH)
    q = """SELECT run_id, pca_k, seed, subject_idx, layer, score FROM results
           WHERE reconstruct_from_pcs = 1 AND cfg_id = ?
             AND analysis = 'rsa' AND compare_method = 'spearman'
             AND neural_dataset = ?"""
    params = [cfg_id, neural_dataset]
    if checkpoint_dir is not None:
        q += " AND checkpoint_dir = ?"
        params.append(checkpoint_dir)
    rfrag, rparams = _region_filter(region)
    df = pd.read_sql(q + rfrag, conn, params=params + rparams)
    conn.close()

    if df.empty:
        return df
    return df.loc[df.groupby(["pca_k", "seed", "subject_idx"])["score"].idxmax()].reset_index(drop=True)


def query_1000way_baseline(neural_dataset, region=None):
    rfrag, rparams = _region_filter(region)
    return _query_best_scores(
        "cfg_id = 1000 AND reconstruct_from_pcs = 0 AND analysis = 'rsa'"
        " AND compare_method = 'spearman' AND neural_dataset = ?" + rfrag,
        [neural_dataset] + rparams,
    )


def query_untrained_baseline(neural_dataset, region=None):
    rfrag, rparams = _region_filter(region)
    return _query_best_scores(
        "cfg_id = 1000 AND epoch = 0 AND reconstruct_from_pcs = 0 AND analysis = 'rsa'"
        " AND compare_method = 'spearman' AND neural_dataset = ?" + rfrag,
        [neural_dataset] + rparams,
    )


def aggregate_curve(df):
    """Mean + bootstrap 95% CI per pca_k."""
    if df.empty:
        return pd.DataFrame(columns=["pca_k", "mean", "ci_low", "ci_high"])
    rows = []
    for pca_k, group in df.groupby("pca_k"):
        _, ci_low, ci_high = get_bootstrap_ci(group["run_id"].tolist())
        rows.append({"pca_k": pca_k, "mean": group["score"].mean(),
                      "ci_low": ci_low, "ci_high": ci_high})
    return pd.DataFrame(rows)


# ── Panel drawing ────────────────────────────────────────────────────────────

def _draw_baseline(ax, baseline, color, linestyle, linewidth, label, zorder=1):
    """Draw a horizontal baseline with optional CI band."""
    mean, ci_lo, ci_hi = baseline
    if np.isnan(mean):
        return
    if not np.isnan(ci_lo):
        ax.axhspan(ci_lo, ci_hi, color=color, alpha=0.12, zorder=0)
    ax.axhline(mean, color=color, linestyle=linestyle,
               linewidth=linewidth, label=label, zorder=zorder)


def plot_panel(ax, curve_df, comparison_baseline,
               untrained_baseline, title, comparison_label="Best coarse model",
               show_ylabel=True):
    """Plot one reconstruction curve panel with a flat comparison baseline."""
    k, mean = curve_df["pca_k"].values, curve_df["mean"].values
    ci_low, ci_high = curve_df["ci_low"].values, curve_df["ci_high"].values

    _draw_baseline(ax, untrained_baseline, UNTRAINED_COLOR, ":", 1.3, "Untrained")

    ax.fill_between(k, ci_low, ci_high, color=FINE_COLOR, alpha=0.18, zorder=2)
    ax.plot(k, mean, "-o", color=FINE_COLOR, markersize=4, linewidth=1.6,
            markeredgecolor="white", markeredgewidth=0.6,
            label="1000-way (top-$k$ PCs)", zorder=3)

    _draw_baseline(ax, comparison_baseline, COARSE_COLOR, "-", 1.4, comparison_label)

    _format_panel(ax, k, title, show_ylabel)


def plot_dual_curves(ax, fine_df, coarse_df, untrained_baseline, title,
                     coarse_label="Coarse model", show_ylabel=True):
    """Plot two reconstruction curves (1000-way and coarse) with untrained baseline."""
    _draw_baseline(ax, untrained_baseline, UNTRAINED_COLOR, ":", 1.3, "Untrained")

    # 1000-way curve
    k = fine_df["pca_k"].values
    ax.fill_between(k, fine_df["ci_low"].values, fine_df["ci_high"].values,
                    color=FINE_COLOR, alpha=0.18, zorder=2)
    ax.plot(k, fine_df["mean"].values, "-o", color=FINE_COLOR, markersize=4,
            linewidth=1.6, markeredgecolor="white", markeredgewidth=0.6,
            label="1000-way (top-$k$ PCs)", zorder=3)

    # Coarse curve
    k_c = coarse_df["pca_k"].values
    ax.fill_between(k_c, coarse_df["ci_low"].values, coarse_df["ci_high"].values,
                    color=COARSE_COLOR, alpha=0.18, zorder=2)
    ax.plot(k_c, coarse_df["mean"].values, "-s", color=COARSE_COLOR, markersize=4,
            linewidth=1.6, markeredgecolor="white", markeredgewidth=0.6,
            label=f"{coarse_label} (top-$k$ PCs)", zorder=3)

    _format_panel(ax, k, title, show_ylabel)


def _format_panel(ax, k, title, show_ylabel):
    """Shared axis formatting for reconstruction panels."""
    ax.set_xlabel("Number of PCs ($k$)", fontsize=9, labelpad=4)
    if show_ylabel:
        ax.set_ylabel(r"Spearman $\rho$", fontsize=9, labelpad=4)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xticks(k)
    labeled = {1, 5, 10, 20, 30, 40, 50} | {int(k[0]), int(k[-1])}
    ax.set_xticklabels(
        [str(int(v)) if int(v) in labeled else "" for v in k], fontsize=8,
    )
    ax.tick_params(axis="both", which="major", labelsize=8,
                   length=TICK_MAJOR_LEN, width=0.8, direction="out")
    ax.yaxis.set_minor_locator(plt.matplotlib.ticker.AutoMinorLocator(2))
    ax.tick_params(axis="y", which="minor", length=TICK_MINOR_LEN, width=0.6, direction="out")
    ax.yaxis.grid(True, which="major", color="#EBEBEB", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, right=True, top=True, offset=4)
    ax.spines["bottom"].set_linewidth(SPINE_WIDTH)
    ax.spines["left"].set_linewidth(SPINE_WIDTH)


# ── Figure assembly ──────────────────────────────────────────────────────────

def plot_figure(neural_dataset, query_comparison_fn, comparison_label,
                filename_prefix, **query_kwargs):
    """Produce a multi-panel figure with a single curve + flat comparison baseline.

    query_comparison_fn(neural_dataset, region, **query_kwargs) → (mean, ci_lo, ci_hi)
    """
    layout = DATASET_LAYOUTS[neural_dataset]
    regions, figsize = layout["regions"], layout["figsize"]
    sns.set_theme(style="ticks", context="paper", font_scale=1.1, rc=RC)

    fig, axes = plt.subplots(1, len(regions), figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i, (region, label) in enumerate(regions):
        curve_df = query_reconstruction_curve(neural_dataset, region)
        agg = aggregate_curve(curve_df)

        if agg.empty:
            axes[i].text(0.5, 0.5, "No data", ha="center", va="center",
                         transform=axes[i].transAxes, fontsize=12, color="#888")
            axes[i].set_title(label, fontsize=11, fontweight="bold")
            continue

        plot_panel(
            axes[i], agg,
            comparison_baseline=query_comparison_fn(neural_dataset, region, **query_kwargs),
            untrained_baseline=query_untrained_baseline(neural_dataset, region),
            title=label, comparison_label=comparison_label, show_ylabel=(i == 0),
        )

    _finish_figure(fig, axes, filename_prefix, neural_dataset)


def plot_dual_figure(neural_dataset, coarse_config, filename_prefix,
                     coarse_label="Coarse model"):
    """Produce a multi-panel figure with two reconstruction curves per panel.

    coarse_config: dict mapping region -> (cfg_id, checkpoint_dir)
    """
    layout = DATASET_LAYOUTS[neural_dataset]
    regions, figsize = layout["regions"], layout["figsize"]
    sns.set_theme(style="ticks", context="paper", font_scale=1.1, rc=RC)

    fig, axes = plt.subplots(1, len(regions), figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i, (region, label) in enumerate(regions):
        # 1000-way reconstruction curve
        fine_df = query_reconstruction_curve(neural_dataset, region)
        fine_agg = aggregate_curve(fine_df)

        # Coarse reconstruction curve (region-specific model)
        cfg_id, checkpoint_dir = coarse_config[region]
        coarse_df = query_reconstruction_curve(
            neural_dataset, region, cfg_id=cfg_id, checkpoint_dir=checkpoint_dir,
        )
        coarse_agg = aggregate_curve(coarse_df)

        if fine_agg.empty and coarse_agg.empty:
            axes[i].text(0.5, 0.5, "No data", ha="center", va="center",
                         transform=axes[i].transAxes, fontsize=12, color="#888")
            axes[i].set_title(label, fontsize=11, fontweight="bold")
            continue

        plot_dual_curves(
            axes[i], fine_agg, coarse_agg,
            untrained_baseline=query_untrained_baseline(neural_dataset, region),
            title=label, coarse_label=coarse_label, show_ylabel=(i == 0),
        )

    _finish_figure(fig, axes, filename_prefix, neural_dataset)


def _finish_figure(fig, axes, filename_prefix, neural_dataset):
    """Shared legend + save logic."""
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        leg = fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8,
                         frameon=True, edgecolor="#cccccc", fancybox=False,
                         bbox_to_anchor=(0.5, -0.01), handlelength=2.2, columnspacing=1.6)
        leg.get_frame().set_linewidth(0.6)

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    out_path = FIGURES_DIR / f"{filename_prefix}_{neural_dataset}.png"
    plt.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {out_path}")
