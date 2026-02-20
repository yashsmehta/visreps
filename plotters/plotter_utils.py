from typing import List, Optional
import json
import sqlite3

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter
from scipy import stats

DB_PATH = "results.db"


# --------------------------------------------------
# SQLite helpers: query scores and bootstrap CIs
# --------------------------------------------------
def query_best_scores(neural_dataset, region, pca_labels_folder, cfg_id,
                      compare_method="spearman", epoch=None, db_path=DB_PATH):
    """Get best-layer score per (seed, subject) from results.db.

    When multiple runs exist for the same (seed, subject), keeps the one
    with the highest score (handles duplicate evaluations).

    Returns DataFrame with columns: seed, subject_idx, score, run_id, layer.
    """
    conn = sqlite3.connect(db_path)
    q = """
    SELECT run_id, seed, subject_idx, layer, score
    FROM results
    WHERE neural_dataset = ? AND region = ? AND pca_labels_folder = ?
      AND cfg_id = ? AND compare_method = ?
    """
    params = [neural_dataset, region, pca_labels_folder, cfg_id, compare_method]
    if epoch is not None:
        q += " AND epoch = ?"
        params.append(epoch)
    df = pd.read_sql(q, conn, params=params)
    conn.close()

    if df.empty:
        return df

    # Keep best score per (seed, subject_idx) — handles duplicate runs
    idx = df.groupby(["seed", "subject_idx"])["score"].idxmax()
    return df.loc[idx].reset_index(drop=True)


def get_bootstrap_ci(run_ids, compare_method="spearman", alpha=0.05, db_path=DB_PATH):
    """Compute bootstrap CI by averaging distributions element-wise across runs.

    For each bootstrap iteration i, averages score_i across all run_ids.
    Returns (mean_score, ci_low, ci_high) where mean_score is the mean of
    the averaged bootstrap distribution.

    Parameters
    ----------
    run_ids : list[str]
        Run IDs to aggregate.
    compare_method : str
        "spearman" or "kendall".
    alpha : float
        Significance level (0.05 → 95% CI).

    Returns
    -------
    tuple[float, float, float]
        (mean, ci_low, ci_high). Returns (nan, nan, nan) if no bootstrap data.
    """
    if not run_ids:
        return np.nan, np.nan, np.nan

    conn = sqlite3.connect(db_path)
    placeholders = ",".join("?" for _ in run_ids)
    q = f"""
    SELECT scores FROM bootstrap_distributions
    WHERE run_id IN ({placeholders}) AND compare_method = ?
    """
    rows = conn.execute(q, list(run_ids) + [compare_method]).fetchall()
    conn.close()

    if not rows:
        return np.nan, np.nan, np.nan

    arrays = [np.array(json.loads(r[0])) for r in rows]

    # Truncate to minimum length (most are 1000, some older runs are 100)
    min_len = min(len(a) for a in arrays)
    arrays = [a[:min_len] for a in arrays]

    # Element-wise mean across runs → distribution of the population mean
    mean_dist = np.mean(arrays, axis=0)

    lo = np.percentile(mean_dist, 100 * alpha / 2)
    hi = np.percentile(mean_dist, 100 * (1 - alpha / 2))
    return float(np.mean(mean_dist)), float(lo), float(hi)


def get_condition_summary(neural_dataset, region, pca_labels_folder, cfg_id,
                          compare_method="spearman", epoch=None, db_path=DB_PATH):
    """Get point estimate + bootstrap 95% CI for one condition.

    Point estimate = mean score across all seeds and subjects (best layer per run).
    CI = from bootstrap distributions averaged across runs.

    Returns dict with keys: mean, ci_low, ci_high, n_runs, run_ids.
    """
    df = query_best_scores(neural_dataset, region, pca_labels_folder, cfg_id,
                           compare_method, epoch, db_path)
    if df.empty:
        return {"mean": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "n_runs": 0, "run_ids": []}

    run_ids = df["run_id"].tolist()
    mean_score = df["score"].mean()
    _, ci_low, ci_high = get_bootstrap_ci(run_ids, compare_method, db_path=db_path)

    # SEM fallback: bootstrap CIs missing, or partial bootstrap that doesn't
    # bracket the point estimate (happens when only some runs have bootstrap)
    needs_fallback = (np.isnan(ci_low)
                      or ci_low > mean_score
                      or ci_high < mean_score)
    if needs_fallback:
        seed_means = df.groupby("seed")["score"].mean()
        if len(seed_means) > 1:
            sem = seed_means.std() / np.sqrt(len(seed_means))
            ci_low = mean_score - 1.96 * sem
            ci_high = mean_score + 1.96 * sem
        else:
            ci_low, ci_high = np.nan, np.nan

    return {"mean": mean_score, "ci_low": ci_low, "ci_high": ci_high,
            "n_runs": len(df), "run_ids": run_ids}


def get_subject_scores(neural_dataset, region, pca_labels_folder, cfg_id,
                       compare_method="spearman", epoch=None, db_path=DB_PATH):
    """Get per-subject scores (averaged across seeds) for box/dot plots.

    Returns Series indexed by subject_idx with mean score per subject.
    """
    df = query_best_scores(neural_dataset, region, pca_labels_folder, cfg_id,
                           compare_method, epoch, db_path)
    if df.empty:
        return pd.Series(dtype=float)

    return df.groupby("subject_idx")["score"].mean()

# --------------------------------------------------
# helpers for skipping / retaining common columns
# --------------------------------------------------
_SKIP_ALWAYS = {"log_interval", "checkpoint_interval", "cfg_id", "score"}
_PCA_COLS   = ("pca_labels", "pca_n_classes")

# --------------------------------------------------
# average across SUBJECT_IDX  (retain SEED if present)
# --------------------------------------------------
def avg_over_subject_idx(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse `subject_idx`; keep `seed` (if any) and PCA columns.
    """
    if df.empty or "subject_idx" not in df:
        return df.copy()

    d = df.copy()
    d["subject_idx"] = pd.to_numeric(d["subject_idx"], errors="coerce")
    d = d.dropna(subset=["subject_idx"])
    if d.empty:
        return d

    skip = _SKIP_ALWAYS | {"subject_idx"}
    group_cols = [c for c in d.columns if c not in skip]

    out = (
        d.groupby(group_cols, dropna=False, observed=False)["score"]
          .mean()
          .reset_index()
    )

    keep = ["layer", "score"]
    if "seed" in out.columns:
        keep.append("seed")
    keep += [c for c in _PCA_COLS if c in out.columns]
    return out[keep]

# --------------------------------------------------
# average across SEED  (retain SUBJECT_IDX if present)
# --------------------------------------------------
def avg_over_seed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse `seed`; keep `subject_idx` (if any) and PCA columns.
    Drop subject_idx if all values are NaN.
    """
    if df.empty or "seed" not in df:
        return df.copy()

    d = df.copy()
    d["seed"] = pd.to_numeric(d["seed"], errors="coerce")
    d = d.dropna(subset=["seed"])
    if d.empty:
        return d

    skip = _SKIP_ALWAYS | {"seed"}
    group_cols = [c for c in d.columns if c not in skip]

    out = (
        d.groupby(group_cols, dropna=False, observed=False)["score"]
          .mean()
          .reset_index()
    )

    keep = ["layer", "score"]
    if "subject_idx" in out.columns:
        if not out["subject_idx"].isna().all():
            keep.append("subject_idx")
    keep += [c for c in _PCA_COLS if c in out.columns]
    return out[keep]

# --------------------------------------------------
# average across SUBJECT_IDX and SEED
# --------------------------------------------------
def avg_over_subject_idx_seed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse `subject_idx` and `seed`; keep PCA columns.
    """
    df_avg_subj = avg_over_subject_idx(df)
    df_avg_subj_seed = avg_over_seed(df_avg_subj)
    return df_avg_subj_seed

# --------------------------------------------------
# split and select df
# --------------------------------------------------
def split_and_select_df(
    df: pd.DataFrame,
    *,
    epoch: Optional[int] = None,
    dataset: Optional[str] = None,
    metric: Optional[str] = None,
    region: Optional[str] = None,
    subject_idx: Optional[List[int]] = None,
    layers: Optional[List[str]] = None,
    pca_n_classes: Optional[List[int]] = None,
    reconstruct_from_pcs: Optional[bool] = None,
    pca_k: Optional[int] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split `df` into two frames:
        - pca_df  – rows with pca_labels == True
        - full_df – rows with pca_labels == False

    Each optional argument filters rows if provided (None -> no filter).
    """
    mask = pd.Series(True, index=df.index)

    if dataset is not None:
        mask &= df["neural_dataset"].str.lower() == dataset.lower()
    if metric is not None:
        mask &= df["compare_rsm_correlation"] == metric
    if region is not None:
        mask &= df["region"] == region
    if epoch is not None:
        mask &= df["epoch"] == epoch
    if subject_idx is not None:
        mask &= df["subject_idx"].isin(subject_idx)
    if layers is not None:
        mask &= df["layer"].isin(layers)
    if pca_n_classes is not None:
        mask &= df["pca_n_classes"].isin(pca_n_classes)
    if reconstruct_from_pcs is not None:
        mask &= df["reconstruct_from_pcs"] == reconstruct_from_pcs
    if pca_k is not None:
        mask &= df["pca_k"] == pca_k

    filt = df[mask].copy()
    flag = filt["pca_labels"].astype(str).str.lower()

    pca_df  = filt[flag.eq("true") ].copy()
    full_df = filt[flag.eq("false")].copy()

    print(f"split_and_select_df: PCA rows : {len(pca_df)}, Full rows: {len(full_df)}\n")

    return pca_df, full_df


# --------------------------------------------------
# best layer selection
# --------------------------------------------------
def get_best_layer_scores(df, group_cols):
    """
    For each unique combination of group_cols, find the layer with highest mean score.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: layer, score, and group_cols
    group_cols : list[str]
        Column names to group by (e.g., ['pca_n_classes'] or ['subject_idx', 'pca_n_classes'])

    Returns
    -------
    dict
        Maps group_key -> (scores_list, best_layer_name)
    """
    result = {}
    for group_vals, group_df in df.groupby(group_cols):
        if len(group_cols) == 1:
            group_vals = group_vals[0] if isinstance(group_vals, tuple) else group_vals

        layer_means = group_df.groupby('layer')['score'].mean()
        best_layer = layer_means.idxmax()

        best_layer_df = group_df[group_df['layer'] == best_layer]
        scores = best_layer_df['score'].tolist()

        result[group_vals] = (scores, best_layer)

    return result


# --------------------------------------------------
# bar plot
# --------------------------------------------------
def plot_brain_score_barplot(scores_by_arch_class, pca_classes, architectures,
                             region_name, out_png, enable_significance=True, ylabel="Brain Similarity (RSA)"):
    """
    Grouped bar-plot for alignment scores across architectures and class counts.

    Parameters
    ----------
    scores_by_arch_class : dict[tuple, list[float]]
        Keys = (architecture, n_classes) tuples, e.g., ('alexnet', 2), ('1K', None).
        Values = list of scores (multiple subjects or single score).
    pca_classes : list[int]
        Ordered list of PCA class counts, e.g., [2, 4, 8, ..., 64].
    architectures : list[str]
        List of architectures to plot (e.g., ['alexnet', 'dino', 'clip']).
    region_name : str
        Brain region or analysis name to display as title.
    out_png : str
        Destination PNG filename.
    enable_significance : bool
        If True, perform paired t-tests vs 1K baseline (requires multiple scores).
    ylabel : str
        Y-axis label (default: "Brain Similarity (RSA)").
    """
    color_map = {
        'alexnet': '#1f77b4',
        'vit': '#ee854a',
        'dino': '#ff7f0e',
        'clip': '#2d7f2d',
        'dreamsim': '#9467bd',
    }
    k1k_color = '#666666'

    sns.set_theme(style='ticks', context='paper', font_scale=1.2)
    fig, ax = plt.subplots(figsize=(16, 6))

    n_archs = len(architectures)
    bar_width = 0.24
    intra_group_gap = 0.04
    group_gap = 0.30

    scores_1k = scores_by_arch_class.get(('1K', None), None)

    # Plot grouped bars with optional significance tests
    for i, n_cls in enumerate(pca_classes):
        base_pos = i * (n_archs * bar_width + (n_archs - 1) * intra_group_gap + group_gap)

        for arch_idx, arch in enumerate(architectures):
            if (arch, n_cls) in scores_by_arch_class:
                scores = scores_by_arch_class[(arch, n_cls)]
                mean_val = np.mean(scores)

                bar_pos = base_pos + arch_idx * (bar_width + intra_group_gap)

                rect = mpatches.FancyBboxPatch(
                    (bar_pos, 0), bar_width, mean_val,
                    boxstyle=mpatches.BoxStyle('Round', pad=.02, rounding_size=.08),
                    facecolor=color_map[arch], edgecolor='black',
                    linewidth=1.0, mutation_aspect=.05
                )
                ax.add_patch(rect)

                # Significance test (only if enabled and sufficient data)
                if enable_significance and scores_1k is not None and len(scores) == len(scores_1k) and len(scores) > 1:
                    t_stat, p_val = stats.ttest_rel(scores, scores_1k)
                    if p_val < 0.01:
                        ax.text(bar_pos + bar_width/2, 0.015, '*',
                               ha='center', va='bottom', fontsize=18, fontweight='bold', color='white')

    # Plot 1K baseline as horizontal line
    if ('1K', None) in scores_by_arch_class:
        scores_1k = scores_by_arch_class[('1K', None)]
        mean_1k = np.mean(scores_1k)
        ax.axhline(y=mean_1k, color=k1k_color, linestyle='--', linewidth=2.5,
                   label='ImageNet-1K', zorder=2, alpha=0.9)

    # X-axis formatting
    tick_positions = []
    tick_labels = []

    for i, n_cls in enumerate(pca_classes):
        base_pos = i * (n_archs * bar_width + (n_archs - 1) * intra_group_gap + group_gap)
        group_width = n_archs * bar_width + (n_archs - 1) * intra_group_gap
        group_center = base_pos + group_width / 2
        tick_positions.append(group_center)
        tick_labels.append(str(n_cls))

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontweight='bold')
    ax.tick_params(axis='x', direction='out', bottom=True, top=False, length=5, color='black',
                   width=1.5, pad=8, labelsize=16)

    # Y-axis formatting
    ax.tick_params(axis='y', which='major', direction='out', left=True, right=False,
                   labelsize=13, length=6, color='black', width=1.5, pad=6)
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    def hide_zero_formatter(x, pos):
        return '' if np.isclose(x, 0) else f'{x:.2f}'
    ax.yaxis.set_major_formatter(FuncFormatter(hide_zero_formatter))

    ax.tick_params(axis='y', which='minor', direction='out', left=True, right=False,
                   length=3, color='black', width=1.0)

    # Set limits
    all_means = [np.mean(v) for v in scores_by_arch_class.values() if len(v) > 0]
    current_max_y = max(all_means) if all_means else 0
    ax.set_ylim(0, current_max_y + 0.025 if current_max_y > 0 else 0.1)

    max_pos = (len(pca_classes) - 1) * (n_archs * bar_width + (n_archs - 1) * intra_group_gap + group_gap)
    max_pos += n_archs * bar_width + (n_archs - 1) * intra_group_gap + 0.5
    ax.set_xlim(-0.5, max_pos)
    ax.set_ylabel(ylabel, fontsize=15, labelpad=12, fontweight='normal')

    ax.set_title(region_name.title(), fontsize=18, fontweight='bold', pad=15)

    # Legend
    legend_handles = []
    arch_name_map = {
        'alexnet': 'AlexNet',
        'vit': 'ViT',
        'dino': 'DINO',
        'clip': 'CLIP',
        'dreamsim': 'DreamSim',
    }
    for arch in architectures:
        arch_display = arch_name_map.get(arch, arch.capitalize())
        patch = mpatches.Patch(facecolor=color_map[arch], edgecolor='black',
                               linewidth=1.0, label=f'{arch_display} classes')
        legend_handles.append(patch)

    k1k_line = mlines.Line2D([], [], color=k1k_color, linestyle='--',
                             linewidth=2.5, label='ImageNet-1K')
    legend_handles.append(k1k_line)

    ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5),
             frameon=True, fontsize=14, framealpha=0.95, edgecolor='black', fancybox=False)

    sns.despine(right=True, top=True, offset=8)
    ax.spines['bottom'].set_linewidth(1.8)
    ax.spines['left'].set_linewidth(1.8)

    plt.tight_layout(pad=1.2, rect=[0, 0, 0.85, 1])
    plt.savefig(out_png, dpi=600, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"Plot saved -> {out_png}")
