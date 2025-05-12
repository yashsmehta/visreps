"""Comparison plots helper + CLI

Refactored, but now restores the original (working) filter logic while
keeping everything else tidy.
"""

from __future__ import annotations

import os
from typing import List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# PLOTTER ----------------------------------------------------------------------
# -----------------------------------------------------------------------------

def create_comparison_plots(
    full_df_filtered: pd.DataFrame,
    pca_df_filtered: pd.DataFrame,
    layer_order: List[str],
    compare_rsm_correlation: str,
    # neural_dataset, region_filter_val, subject_filter_val are removed as this is THINGS-specific
):
    """Return saved‑figure path for a THINGS dataset comparison plot."""

    # --- Config & Style (THINGS specific) ------------------------------------
    pca_sizes = [2, 4, 8, 16, 32, 64]
    
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 13,
        'lines.linewidth': 1.75,
        'lines.markersize': 6,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.alpha': 0.5
    })

    colours = {
        "initial": "#7f8c8d", 
        "final": "#c0392b",   
        "pca": dict(zip(pca_sizes, plt.cm.Blues(np.linspace(0.2, 0.8, len(pca_sizes))))),
    }

    # --- Data Aggregation (THINGS specific - no subject/region averaging) ----
    # For THINGS, we assume data is already representative or doesn't need subject/region grouping.
    # We directly aggregate by layer, epoch, and pca_n_classes.
    
    base_cols_for_agg = ["layer", "epoch"]
    pca_base_cols_for_agg = ["layer", "epoch", "pca_labels", "pca_n_classes"]

    def _agg(df: pd.DataFrame, cols_for_groupby):
        # Simplified aggregation for THINGS: just mean of scores
        if df.empty:
            return pd.DataFrame(columns=cols_for_groupby + ['score'])
        return df.groupby(cols_for_groupby, observed=False)["score"].mean().reset_index()

    full_agg = _agg(full_df_filtered, base_cols_for_agg)
    pca_agg  = _agg(pca_df_filtered, pca_base_cols_for_agg)
    
    # --- Determine Global Y-axis Limits ---
    all_scores_for_ylim = []
    initial_scores = full_agg[full_agg["epoch"] == 0]["score"]
    if not initial_scores.empty: all_scores_for_ylim.append(initial_scores)
    
    final_scores = full_agg[full_agg["epoch"] == 18]["score"]
    if not final_scores.empty: all_scores_for_ylim.append(final_scores)

    for n_classes in pca_sizes:
        pca_n_scores = pca_agg[(pca_agg["pca_n_classes"] == n_classes) & (pca_agg["epoch"] == 18)]["score"]
        if not pca_n_scores.empty: all_scores_for_ylim.append(pca_n_scores)
    
    min_y, max_y = None, None
    if all_scores_for_ylim:
        global_scores_series = pd.concat(all_scores_for_ylim).dropna()
        if not global_scores_series.empty:
            min_y = global_scores_series.min()
            max_y = global_scores_series.max()
            padding = (max_y - min_y) * 0.05 if (max_y - min_y) > 1e-9 else 0.05
            min_y -= padding
            max_y += padding

    # --- Plotting (Single plot for THINGS) ---
    fig, ax = plt.subplots(1, 1, figsize=(6, 5.5)) # Single subplot

    x_ticks_pos = range(len(layer_order))

    # Plot initial (full model, epoch 0)
    initial_data = full_agg[full_agg["epoch"] == 0].set_index("layer").reindex(layer_order)["score"]
    if not initial_data.empty and not initial_data.isna().all():
        ax.plot(x_ticks_pos, initial_data, color=colours["initial"], marker="x", linestyle="-", label="Initial", zorder=3)
    
    # Plot final (full model, epoch 18)
    final_data = full_agg[full_agg["epoch"] == 18].set_index("layer").reindex(layer_order)["score"]
    if not final_data.empty and not final_data.isna().all():
        ax.plot(x_ticks_pos, final_data, color=colours["final"], marker="o", linestyle="-", label="1000 Classes", zorder=3)

    # Plot PCA variants (epoch 18)
    pca_lines_data = {}
    for n_classes in pca_sizes:
        pca_lines_data[n_classes] = pca_agg[
            (pca_agg["pca_n_classes"] == n_classes) & (pca_agg["epoch"] == 18)
        ].set_index("layer").reindex(layer_order)["score"]

    for n_classes, pca_series in pca_lines_data.items():
        if not pca_series.empty and not pca_series.isna().all():
            ax.plot(x_ticks_pos, pca_series, color=colours["pca"][n_classes], marker="s", linestyle="-", label=f"PCA {n_classes} classes", zorder=2)

    ax.set_xticks(x_ticks_pos)
    ax.set_xticklabels(layer_order, rotation=45, ha="right")
    ax.set_title("THINGS Dataset Comparison") # Static title
    ax.set_ylabel(f"RSA ({compare_rsm_correlation.capitalize()})")
    
    if min_y is not None and max_y is not None:
        ax.set_ylim(min_y, max_y)

    # --- Figure-level Adjustments ---
    # Legend
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=min(len(handles), 4), frameon=False)
        plt.tight_layout(rect=[0, 0.08, 1, 0.95]) # Adjusted top for title
    else:
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

    # --- Save Figure ---
    out_dir = "plotters/plots"
    os.makedirs(out_dir, exist_ok=True)
    
    # Simplified filename for THINGS
    fname = f"comparison_things_{compare_rsm_correlation.lower()}.png"
    
    path = os.path.join(out_dir, fname)
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    return [path]

# -----------------------------------------------------------------------------
# CLI -------------------------------------------------------------------------
# -----------------------------------------------------------------------------

def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    print("  •", path)
    return pd.read_csv(path)


def main():
    cfg = {
        # "dataset": "things", # Implicitly THINGS for this plotter
        "metric": "Spearman",        # 'Kendall', 'Spearman', ...
        # "region": "N/A",       # Not used for THINGS specific plotter
        # "subject": "N/A",      # Not used for THINGS specific plotter
        "layer_order": ["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2"],
        "log_dir": "logs", 
        "combined_csv": "full_vs_pcs_things.csv",
    }

    print("\n*** Plot generation cfg (THINGS specific) ***")
    print(f"  metric: {cfg['metric']}")
    print(f"  log_dir: {cfg['log_dir']}")
    print(f"  combined_csv: {cfg['combined_csv']}")
    # layer_order is long, so not printed by default

    # --- Load ----------------------------------------------------------------
    combined_raw = _load_csv(os.path.join(cfg["log_dir"], cfg["combined_csv"]))

    if "pca_labels" not in combined_raw.columns:
        raise ValueError("'pca_labels' column not found in the combined CSV file.")

    pca_labels_as_str_lower = combined_raw["pca_labels"].astype(str).str.strip().str.lower()
    full_raw = combined_raw[pca_labels_as_str_lower == "false"].copy()
    pca_raw  = combined_raw[pca_labels_as_str_lower == "true"].copy()

    # --- Apply Filters (Simplified for THINGS) -------------------------------
    metric = cfg["metric"].strip()
    print("\nApplying filters…")

    # Dataset filter (neural_dataset column must be 'things')
    # This also implicitly filters out any data not related to 'things' if the CSV is mixed.
    things_ds_key = "things"
    if "neural_dataset" in full_raw.columns:
        full = full_raw[full_raw["neural_dataset"].astype(str).str.strip().str.lower() == things_ds_key].copy()
    else:
        print("  Warning: 'neural_dataset' column missing in full_raw. Assuming all data is for THINGS.")
        full = full_raw.copy() # Or handle as an error if strictly 'things' data is expected

    if "neural_dataset" in pca_raw.columns:
        pca  = pca_raw[pca_raw["neural_dataset"].astype(str).str.strip().str.lower() == things_ds_key].copy()
    else:
        print("  Warning: 'neural_dataset' column missing in pca_raw. Assuming all data is for THINGS.")
        pca = pca_raw.copy()
    print(f"  Filtered for neural_dataset='{things_ds_key}': {len(full)} (full), {len(pca)} (pca)")

    # Metric filter
    for df, name in ((full, "full"), (pca, "pca")):
        if "compare_rsm_correlation" in df.columns:
            condition = df["compare_rsm_correlation"].astype(str).str.strip() != metric
            df.drop(df[condition].index, inplace=True)
        else:
            print(f"  Warning: 'compare_rsm_correlation' missing in {name} CSV")
    print(f"  Filtered for metric='{metric}': {len(full)} (full), {len(pca)} (pca)")

    # Region and Subject specific filters are removed as this plotter is THINGS-specific
    # and create_comparison_plots no longer expects/uses region/subject distinctions.

    # --- Validate ------------------------------------------------------------
    if full.empty or pca.empty:
        # Provide more context if empty after filtering for 'things' dataset
        raise ValueError(
            f"Empty dataframe(s) after filtering for dataset '{things_ds_key}' and metric '{metric}'. "
            f"Check CSV content and ensure 'neural_dataset' (as '{things_ds_key}') and 'compare_rsm_correlation' (as '{metric}') columns are present and populated correctly." 
            "Also verify the 'pca_labels' column for correct True/False separation."
        )

    # --- Plot ----------------------------------------------------------------
    paths = create_comparison_plots(
        full,
        pca,
        cfg["layer_order"],
        cfg["metric"],
        # No neural_dataset, region, or subject args passed
    )

    print(f"\nGenerated {len(paths)} plot(s):")
    for p in paths:
        print("  •", p)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("Error:", exc)
