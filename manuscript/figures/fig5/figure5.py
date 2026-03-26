"""Figure 5: Architecture Generalization + Data Efficiency.

Layout (2 rows):
  Top row (A): THINGS coarseness for ResNet-50 | ConvNeXt | ViT-B/16
               (CLIP-based coarse labels, epoch 20, seed 1)
  Bottom row (B): Data efficiency — NSD Early | NSD Ventral | THINGS

Usage:
    python manuscript/figures/fig5/figure5.py
"""

import os
import sys
import json
import sqlite3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.ticker import (
    AutoMinorLocator, FuncFormatter, FixedLocator, NullLocator,
)
from matplotlib.transforms import blended_transform_factory
import seaborn as sns

sys.path.insert(0, "manuscript/figures")
from fig_utils import (
    COARSE_CFGS, MARKER_SIZE, EDGE_COLOR, EDGE_WIDTH, setup_style,
)

# ── Config ────────────────────────────────────────────────────────────────
OUTPUT_DIR = "manuscript/figures/fig5"
DB_PATH = "results.db"
DATA_EFF_CSV = os.path.join("experiments", "coarse_grain_benefits",
                            "data_efficiency", "data_efficiency_results.csv")

# ── Top row: Architecture coarseness panels ──────────────────────────────
ARCH_MODELS = [
    ("ResNet50",      "ResNet-50"),
    ("ConvNeXt_Base", "ConvNeXt"),
    ("ViTBase",       "ViT-B/16"),
]

# Match Figure 2 color scheme
CLIP_STYLE = {"color": "#08519c", "marker": "s"}   # dark blue square
BASELINE_1K_COLOR = "#e8963e"                        # warm amber
BAR_CENTER = 250
BAR_WIDTH_FRAC = 0.15

# ── Bottom row: Data efficiency ──────────────────────────────────────────
DE_COLORS = {
    8:    "#a1d99b",   # light green
    16:   "#41ab5d",   # medium green
    32:   "#006d2c",   # dark green
    1000: "#e6550d",   # vivid orange
}
DE_MARKERS = {8: "o", 16: "s", 32: "D", 1000: "X"}
DE_CONDITIONS = [8, 16, 32, 1000]
DATASETS = ["imagenet-mini-5", "imagenet-mini-10", "imagenet-mini-50",
            "imagenet-full"]
DATASET_LABELS = {
    "imagenet-mini-5": "5K", "imagenet-mini-10": "10K",
    "imagenet-mini-50": "50K", "imagenet-full": "1.2M",
}
BENCHMARKS = {
    "nsd_early": {"title": "NSD (Early Visual Stream)",
                  "ylabel": r"RSA (Spearman $\rho$)"},
    "nsd":       {"title": "NSD (Ventral Visual Stream)",
                  "ylabel": r"RSA (Spearman $\rho$)"},
    "things":    {"title": "THINGS (Behavioral)",
                  "ylabel": r"RSA (Spearman $\rho$)"},
}


# ═══════════════════════════════════════════════════════════════════════════
# Top row — THINGS coarseness per architecture
# ═══════════════════════════════════════════════════════════════════════════

def fetch_things_arch_data(model_name, epoch=20, seed=1):
    """Fetch THINGS-behavior scores for a specific architecture (CLIP labels).

    Returns (coarse_dict, baseline_dict_or_None, untrained_score_or_None).
    coarse_dict: {cfg_id: {score, ci_low, ci_high}} for coarse conditions.
    """
    conn = sqlite3.connect(DB_PATH)
    results = {}

    # ── Coarse conditions (CLIP labels) ──
    for cfg in COARSE_CFGS:
        df = pd.read_sql("""
            SELECT r.run_id, r.score, r.ci_low, r.ci_high,
                   bd.scores AS boot_scores
            FROM results r
            LEFT JOIN bootstrap_distributions bd
                ON r.run_id = bd.run_id AND bd.compare_method = 'spearman'
            WHERE r.neural_dataset = 'things-behavior'
              AND r.model_name = ? AND r.cfg_id = ? AND r.epoch = ?
              AND r.seed = ? AND r.pca_labels_folder = 'pca_labels_clip'
              AND r.compare_method = 'spearman' AND r.reconstruct_from_pcs = 0
            ORDER BY r.score DESC LIMIT 1
        """, conn, params=[model_name, cfg, epoch, seed])
        if not df.empty:
            row = df.iloc[0]
            ci_low, ci_high = row["ci_low"], row["ci_high"]
            if row["boot_scores"] is not None:
                boots = np.array(json.loads(row["boot_scores"]))
                ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
            results[cfg] = {"score": row["score"],
                            "ci_low": ci_low, "ci_high": ci_high}

    # ── 1000-way baseline ──
    df_1k = pd.read_sql("""
        SELECT r.run_id, r.score, r.ci_low, r.ci_high,
               bd.scores AS boot_scores
        FROM results r
        LEFT JOIN bootstrap_distributions bd
            ON r.run_id = bd.run_id AND bd.compare_method = 'spearman'
        WHERE r.neural_dataset = 'things-behavior'
          AND r.model_name = ? AND r.cfg_id = 1000 AND r.epoch = ?
          AND r.seed = ? AND r.compare_method = 'spearman'
          AND r.reconstruct_from_pcs = 0
        ORDER BY r.score DESC LIMIT 1
    """, conn, params=[model_name, epoch, seed])
    baseline = None
    if not df_1k.empty:
        row = df_1k.iloc[0]
        ci_low, ci_high = row["ci_low"], row["ci_high"]
        if row["boot_scores"] is not None:
            boots = np.array(json.loads(row["boot_scores"]))
            ci_low, ci_high = np.percentile(boots, [2.5, 97.5])
        baseline = {"score": row["score"],
                    "ci_low": ci_low, "ci_high": ci_high}

    # ── Untrained (epoch=0) ──
    df_un = pd.read_sql("""
        SELECT score FROM results
        WHERE neural_dataset = 'things-behavior'
          AND model_name = ? AND epoch = 0 AND seed = ?
          AND compare_method = 'spearman' AND reconstruct_from_pcs = 0
        ORDER BY score DESC LIMIT 1
    """, conn, params=[model_name, seed])
    untrained = df_un.iloc[0]["score"] if not df_un.empty else None

    conn.close()
    return results, baseline, untrained


def _draw_bar_break(ax):
    """Draw // break marks between the coarse scatter region and the bar."""
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    mid = np.exp((np.log(64) + np.log(BAR_CENTER)) / 2)
    rect_hw = mid * 0.16
    rect = mpatches.FancyBboxPatch(
        (mid / 1.16, -0.05), width=rect_hw * 1.5, height=0.10,
        boxstyle="square,pad=0", facecolor="white", edgecolor="none",
        transform=trans, clip_on=False, zorder=9)
    ax.add_patch(rect)
    for x_shift in [0.93, 1.07]:
        x_c = mid * x_shift
        ax.plot([x_c / 1.04, x_c * 1.04], [-0.028, 0.028],
                transform=trans, color="#555555", linewidth=0.7,
                clip_on=False, zorder=11)


def plot_things_coarseness(ax, model_name, display_name,
                           show_ylabel=True, show_xlabel=True,
                           forced_ylim=None):
    """Plot THINGS coarseness panel in Figure 2 style for a single architecture."""
    results, baseline, untrained = fetch_things_arch_data(model_name)

    if not results and baseline is None:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="#888")
        return

    all_y_vals = []

    # ── Coarse scatter points ──
    for cfg in COARSE_CFGS:
        if cfg not in results:
            continue
        r = results[cfg]
        all_y_vals.append(r["score"])
        err_lo = max(r["score"] - r["ci_low"], 0) if pd.notna(r["ci_low"]) else 0
        err_hi = max(r["ci_high"] - r["score"], 0) if pd.notna(r["ci_high"]) else 0
        ax.errorbar(cfg, r["score"],
                    yerr=[[err_lo], [err_hi]],
                    fmt=CLIP_STYLE["marker"], color=CLIP_STYLE["color"],
                    markersize=MARKER_SIZE,
                    markeredgecolor=EDGE_COLOR, markeredgewidth=EDGE_WIDTH,
                    capsize=1.5, capthick=0.5,
                    ecolor=CLIP_STYLE["color"], elinewidth=0.7, zorder=4)

    # ── Collect y-range ──
    if baseline:
        all_y_vals.append(baseline["score"])
    if untrained is not None:
        all_y_vals.append(untrained)

    y_min = min(all_y_vals)
    y_max = max(all_y_vals)
    y_range = y_max - y_min if y_max > y_min else 0.05
    y_bottom = y_min - y_range * 0.12

    # ── Untrained dashed line ──
    if untrained is not None:
        ax.axhline(untrained, color="#AAAAAA", linestyle="--",
                   linewidth=0.9, alpha=0.7, zorder=1)
        y_offset = y_range * 0.03
        ax.text(0.02, untrained + y_offset, " Untrained",
                fontsize=6, fontstyle="italic", color="#AAAAAA",
                ha="left", va="bottom",
                transform=blended_transform_factory(ax.transAxes, ax.transData),
                zorder=10)

    # ── 1000-way bar ──
    if baseline:
        bl = baseline
        bl_err_lo = (max(bl["score"] - bl["ci_low"], 0)
                     if bl["ci_low"] is not None and pd.notna(bl["ci_low"]) else 0)
        bl_err_hi = (max(bl["ci_high"] - bl["score"], 0)
                     if bl["ci_high"] is not None and pd.notna(bl["ci_high"]) else 0)
        ax.bar(BAR_CENTER, bl["score"] - y_bottom, bottom=y_bottom,
               width=BAR_CENTER * BAR_WIDTH_FRAC,
               color=BASELINE_1K_COLOR, edgecolor="#c07830",
               linewidth=0.4, zorder=3)
        if bl_err_lo > 0 or bl_err_hi > 0:
            ax.errorbar(BAR_CENTER, bl["score"],
                        yerr=[[bl_err_lo], [bl_err_hi]],
                        fmt="none", ecolor="#555555", elinewidth=0.7,
                        capsize=2.2, capthick=0.6, zorder=5)

    # ── Axis formatting (match Figure 2) ──
    ax.set_xscale("log", base=2)
    all_ticks = COARSE_CFGS + [BAR_CENTER]
    label_map = {v: str(v) for v in COARSE_CFGS}
    label_map[BAR_CENTER] = "1000"

    def _fmt(val, pos):
        for k, lbl in label_map.items():
            if abs(val - k) < k * 0.05:
                return lbl
        return ""

    ax.xaxis.set_major_locator(FixedLocator(all_ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(_fmt))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.tick_params(axis="x", which="major", length=3.5, width=0.6)
    ax.set_xlim(1.5, BAR_CENTER * 1.35)

    ax.tick_params(axis="y", which="major", direction="out", length=3.5,
                   width=0.6)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="minor", direction="out", length=2,
                   width=0.4)
    ax.yaxis.grid(True, which="major", color="#F0F0F0", linewidth=0.3,
                  zorder=0)
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:.2f}".rstrip("0").rstrip(".")))

    if forced_ylim is not None:
        cur_ylim = ax.get_ylim()
        yl = forced_ylim[0] if forced_ylim[0] is not None else cur_ylim[0]
        yh = forced_ylim[1] if forced_ylim[1] is not None else cur_ylim[1] + y_range * 0.03
        ax.set_ylim(yl, yh)
    else:
        cur_ylim = ax.get_ylim()
        ax.set_ylim(cur_ylim[0], cur_ylim[1] + y_range * 0.03)

    if show_xlabel:
        ax.set_xlabel("ImageNet training classes", fontsize=9, labelpad=6)
    if show_ylabel:
        ax.set_ylabel(r"RSA (Spearman $\rho$)", fontsize=9, labelpad=3)
    else:
        ax.set_ylabel("")
    sns.despine(ax=ax, right=True, top=True, offset=3)

    _draw_bar_break(ax)


# ═══════════════════════════════════════════════════════════════════════════
# Bottom row — Data efficiency (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════════

def load_csv_data():
    """Load data-efficiency CSV and aggregate to one row per
    (dataset, condition, benchmark)."""
    df = pd.read_csv(DATA_EFF_CSV)
    rows = []

    for bench in ["nsd", "nsd_early", "things"]:
        bdf = df[df["benchmark"] == bench]
        if bdf.empty:
            continue

        for ds in ["imagenet-mini-5", "imagenet-mini-10", "imagenet-mini-50"]:
            dsdf = bdf[bdf["dataset"] == ds]
            for cond in DE_CONDITIONS:
                cdf = dsdf[dsdf["condition"] == cond]
                if cdf.empty:
                    continue

                if bench in ("nsd", "nsd_early"):
                    epoch_scores = {}
                    for epoch, edf in cdf.groupby("epoch"):
                        epoch_scores[epoch] = {
                            "score": edf["score"].mean(),
                            "ci_low": edf["ci_low"].mean(),
                            "ci_high": edf["ci_high"].mean(),
                        }
                    best_epoch = max(epoch_scores,
                                     key=lambda e: epoch_scores[e]["score"])
                    vals = epoch_scores[best_epoch]
                else:
                    best_row = cdf.loc[cdf["score"].idxmax()]
                    vals = {
                        "score": best_row["score"],
                        "ci_low": best_row["ci_low"],
                        "ci_high": best_row["ci_high"],
                    }

                rows.append({"dataset": ds, "condition": cond,
                             "benchmark": bench, **vals})

    return pd.DataFrame(rows)


def load_full_imagenet():
    """Load full ImageNet (1.2M) results from results.db with bootstrap CIs."""
    conn = sqlite3.connect(DB_PATH)
    rows = []

    for cond in DE_CONDITIONS:
        if cond == 1000:
            where = ("cfg_id=1000 AND model_name='CustomCNN' "
                     "AND reconstruct_from_pcs=0 AND epoch=20")
        else:
            where = (f"cfg_id={cond} AND model_name='CustomCNN' "
                     f"AND pca_labels_folder='pca_labels_clip' "
                     f"AND reconstruct_from_pcs=0 AND epoch=20")

        # ── THINGS ──
        things = pd.read_sql(f"""
            SELECT seed, score, ci_low, ci_high
            FROM results
            WHERE neural_dataset='things-behavior'
              AND compare_method='spearman' AND {where}
            ORDER BY seed, score DESC
        """, conn)
        if not things.empty:
            best = things.groupby("seed").first().reset_index()
            mean_score = best["score"].mean()
            if best["ci_low"].notna().all():
                ci_low = best["ci_low"].mean()
                ci_high = best["ci_high"].mean()
            else:
                sem = best["score"].std() / np.sqrt(len(best))
                ci_low = mean_score - 1.96 * sem
                ci_high = mean_score + 1.96 * sem
            rows.append({
                "dataset": "imagenet-full", "condition": cond,
                "benchmark": "things",
                "score": mean_score, "ci_low": ci_low, "ci_high": ci_high,
            })

        # ── NSD ventral stream ──
        nsd_best = pd.read_sql(f"""
            SELECT run_id, seed, subject_idx, score,
                   ROW_NUMBER() OVER (
                       PARTITION BY seed, subject_idx
                       ORDER BY score DESC) as rn
            FROM results
            WHERE neural_dataset='nsd'
              AND region='ventral visual stream'
              AND compare_method='spearman' AND {where}
        """, conn)
        if nsd_best.empty:
            continue
        nsd_best = nsd_best[nsd_best["rn"] == 1].drop(columns=["rn"])
        mean_score = nsd_best["score"].mean()
        ci_low, ci_high = _bootstrap_ci(conn, nsd_best, mean_score)
        rows.append({
            "dataset": "imagenet-full", "condition": cond,
            "benchmark": "nsd",
            "score": mean_score, "ci_low": ci_low, "ci_high": ci_high,
        })

        # ── NSD early visual stream ──
        nsd_early = pd.read_sql(f"""
            SELECT run_id, seed, subject_idx, score,
                   ROW_NUMBER() OVER (
                       PARTITION BY seed, subject_idx
                       ORDER BY score DESC) as rn
            FROM results
            WHERE neural_dataset='nsd'
              AND region='early visual stream'
              AND compare_method='spearman' AND {where}
        """, conn)
        if not nsd_early.empty:
            nsd_early = nsd_early[nsd_early["rn"] == 1].drop(columns=["rn"])
            early_mean = nsd_early["score"].mean()
            e_ci_low, e_ci_high = _bootstrap_ci(conn, nsd_early, early_mean)
            rows.append({
                "dataset": "imagenet-full", "condition": cond,
                "benchmark": "nsd_early",
                "score": early_mean, "ci_low": e_ci_low,
                "ci_high": e_ci_high,
            })

    conn.close()
    return pd.DataFrame(rows)


def _bootstrap_ci(conn, best_df, mean_score):
    """Compute 95% CI from bootstrap distributions, falling back to SEM."""
    run_ids = best_df["run_id"].unique().tolist()
    placeholders = ",".join(f"'{r}'" for r in run_ids)
    boot_dists = pd.read_sql(f"""
        SELECT bd.run_id, bd.scores
        FROM bootstrap_distributions bd
        WHERE bd.run_id IN ({placeholders})
          AND bd.compare_method='spearman'
    """, conn)

    if not boot_dists.empty:
        boot_dists = boot_dists.merge(
            best_df[["run_id", "seed", "subject_idx"]], on="run_id")
        n_boot = 1000
        seed_boots = []
        for seed, sdf in boot_dists.groupby("seed"):
            arrays = [np.array(json.loads(s)) for s in sdf["scores"].values]
            arrays = [a for a in arrays if len(a) == n_boot]
            if len(arrays) < 2:
                continue
            seed_boots.append(np.vstack(arrays).mean(axis=0))
        if seed_boots:
            all_boots = np.vstack(seed_boots).mean(axis=0)
            return tuple(np.percentile(all_boots, [2.5, 97.5]))

    sem = best_df["score"].std() / np.sqrt(len(best_df))
    return mean_score - 1.96 * sem, mean_score + 1.96 * sem


def plot_de_panel(ax, data, benchmark):
    """Line plot for one data-efficiency benchmark panel."""
    x_positions = np.arange(len(DATASETS))
    x_map = {ds: i for i, ds in enumerate(DATASETS)}

    for cond in DE_CONDITIONS:
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
                    fmt="none", ecolor=DE_COLORS[cond], capsize=3,
                    linewidth=1.0, capthick=0.8, alpha=0.65, zorder=2)
        ax.plot(xs, ys, marker=DE_MARKERS[cond], color=DE_COLORS[cond],
                markersize=7, linewidth=2.0, markeredgecolor="white",
                markeredgewidth=0.7, label=label, zorder=3)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([DATASET_LABELS[ds] for ds in DATASETS], fontsize=9)
    ax.set_xlabel("Training images", fontsize=10, labelpad=6)
    ax.set_ylabel(BENCHMARKS[benchmark]["ylabel"], fontsize=10, labelpad=6)
    ax.set_title(BENCHMARKS[benchmark]["title"], fontsize=11,
                 fontweight="bold", pad=10)
    ax.yaxis.grid(True, which="major", color="#ECECEC", linewidth=0.5,
                  zorder=0)
    ax.set_axisbelow(True)
    ax.margins(x=0.08)
    ax.tick_params(axis="both", which="major", direction="out", length=4,
                   width=0.8)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="minor", direction="out", length=2.5,
                   width=0.5)
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:.2f}".rstrip("0").rstrip(".")))
    sns.despine(ax=ax, right=True, top=True, offset=5)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    setup_style()
    plt.rcParams.update({
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
    })

    # ── Figure layout: single row, 3 panels ──
    fig = plt.figure(figsize=(13, 4.0))
    fig.patch.set_facecolor("white")

    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.28,
                           left=0.07, right=0.97, top=0.82, bottom=0.15)

    # Pre-fetch data to compute shared y-limits for ResNet-50 & ConvNeXt
    arch_data = {}
    for model_name, display_name in ARCH_MODELS:
        arch_data[model_name] = fetch_things_arch_data(model_name)

    # Shared ylim for ResNet-50 and ConvNeXt (indices 0 and 1)
    shared_y_vals = []
    for mn in ["ResNet50", "ConvNeXt_Base"]:
        results, baseline, untrained = arch_data[mn]
        for r in results.values():
            shared_y_vals.extend([r["score"] - (r["score"] - r["ci_low"])
                                  if pd.notna(r["ci_low"]) else r["score"],
                                  r["ci_high"]
                                  if pd.notna(r["ci_high"]) else r["score"]])
        if baseline:
            shared_y_vals.append(baseline["score"])
        if untrained is not None:
            shared_y_vals.append(untrained)

    shared_ymin = min(shared_y_vals)
    shared_ymax = max(shared_y_vals)
    shared_range = shared_ymax - shared_ymin
    shared_ylim = (0.1, shared_ymax + shared_range * 0.08)

    axes = []
    for i, (model_name, display_name) in enumerate(ARCH_MODELS):
        ax = fig.add_subplot(gs[0, i])
        ylim = shared_ylim if i < 2 else (0.1, None)  # All start at 0.1
        plot_things_coarseness(ax, model_name, display_name,
                               show_ylabel=(i == 0), show_xlabel=True,
                               forced_ylim=ylim)
        axes.append(ax)

    # Architecture subtitles above each panel
    for i, (_, display_name) in enumerate(ARCH_MODELS):
        pos = axes[i].get_position()
        x_center = (pos.x0 + pos.x1) / 2
        fig.text(x_center, pos.y1 + 0.012, display_name,
                 fontsize=9, color="#888888",
                 ha="center", va="bottom", family="sans-serif")

    # Legend in first panel — just the coarse label source (CLIP)
    coarse_handle = Line2D([], [], marker=CLIP_STYLE["marker"],
                           color="none",
                           markerfacecolor=CLIP_STYLE["color"],
                           markeredgecolor=EDGE_COLOR,
                           markeredgewidth=EDGE_WIDTH,
                           markersize=5.5, label="CLIP")
    axes[0].legend(handles=[coarse_handle],
                   fontsize=7.5, frameon=True, fancybox=False,
                   framealpha=0.92, edgecolor="#dddddd",
                   borderpad=0.4, handletextpad=0.3,
                   labelspacing=0.25,
                   title="Coarse label source",
                   title_fontsize=7,
                   loc="center left",
                   bbox_to_anchor=(0.0, 0.35))

    # ── Panel labels: A, B, C ──
    top_y = axes[0].get_position().y1
    label_y = top_y + 0.035
    for i, label in enumerate(["A", "B", "C"]):
        pos = axes[i].get_position()
        fig.text(pos.x0 - 0.03, label_y, label,
                 fontsize=14, fontweight="bold", va="bottom", ha="left",
                 family="sans-serif")

    # ── Save ──
    out = f"{OUTPUT_DIR}/figure5.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white",
                edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()
