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

CURVE_COLOR = "#e6a200"       # golden amber — reconstruction curve
TRAINED_COLOR = "#d45500"     # burnt orange-red — 1000-way baseline
COARSE_COLOR = "#2166ac"      # blue — coarse / 2-class models
UNTRAINED_COLOR = "#969696"   # neutral grey

SPINE_WIDTH = 1.3
TICK_MAJOR_LEN = 5
TICK_MINOR_LEN = 3

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


def query_reconstruction_curve(neural_dataset, region=None):
    """Per-(pca_k, seed, subject) best-layer scores for reconstruction runs."""
    conn = sqlite3.connect(DB_PATH)
    q = """SELECT run_id, pca_k, seed, subject_idx, layer, score FROM results
           WHERE reconstruct_from_pcs = 1 AND cfg_id = 1000
             AND analysis = 'rsa' AND compare_method = 'spearman'
             AND neural_dataset = ?"""
    params = [neural_dataset]
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
        ax.axhspan(ci_lo, ci_hi, color=color, alpha=0.08, zorder=0)
    ax.axhline(mean, color=color, linestyle=linestyle,
               linewidth=linewidth, label=label, zorder=zorder)


def plot_panel(ax, curve_df, trained_baseline, comparison_baseline,
               untrained_baseline, title, comparison_label="Best coarse model",
               show_ylabel=True):
    """Plot one reconstruction curve panel with baselines."""
    k, mean = curve_df["pca_k"].values, curve_df["mean"].values
    ci_low, ci_high = curve_df["ci_low"].values, curve_df["ci_high"].values

    _draw_baseline(ax, untrained_baseline, UNTRAINED_COLOR, ":", 1.3, "Untrained")
    _draw_baseline(ax, trained_baseline, TRAINED_COLOR, "--", 1.3, "1000-way (all dims)")

    ax.fill_between(k, ci_low, ci_high, color=CURVE_COLOR, alpha=0.15, zorder=2)
    ax.plot(k, mean, "-o", color=CURVE_COLOR, markersize=3.5, linewidth=1.6,
            markeredgecolor="white", markeredgewidth=0.6,
            label="1000-way (top-$k$ PCs)", zorder=3)

    _draw_baseline(ax, comparison_baseline, COARSE_COLOR, "-", 1.4, comparison_label)

    # Formatting
    ax.set_xlabel("Number of PCs ($k$)", fontsize=10, labelpad=4)
    if show_ylabel:
        ax.set_ylabel("Spearman $\\rho$", fontsize=10, labelpad=4)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xticks(k)
    ax.set_xticklabels([str(int(v)) if v % 2 == 1 or v == 2 else "" for v in k], fontsize=8)
    ax.tick_params(axis="both", which="major", labelsize=8.5,
                   length=TICK_MAJOR_LEN, width=0.8, direction="out")
    ax.yaxis.set_minor_locator(plt.matplotlib.ticker.AutoMinorLocator(2))
    ax.tick_params(axis="y", which="minor", length=TICK_MINOR_LEN, width=0.6, direction="out")
    ax.yaxis.grid(True, linestyle="-", alpha=0.15, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, right=True, top=True, offset=4)
    ax.spines["bottom"].set_linewidth(SPINE_WIDTH)
    ax.spines["left"].set_linewidth(SPINE_WIDTH)


# ── Figure assembly ──────────────────────────────────────────────────────────

def plot_figure(neural_dataset, query_comparison_fn, comparison_label,
                filename_prefix, **query_kwargs):
    """Produce a multi-panel figure for one neural dataset.

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
            trained_baseline=query_1000way_baseline(neural_dataset, region),
            comparison_baseline=query_comparison_fn(neural_dataset, region, **query_kwargs),
            untrained_baseline=query_untrained_baseline(neural_dataset, region),
            title=label, comparison_label=comparison_label, show_ylabel=(i == 0),
        )

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
