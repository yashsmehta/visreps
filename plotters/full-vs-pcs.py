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
    neural_dataset: str,
    compare_rsm_correlation: str,
    region_filter_val: str,
    subject_filter_val: Union[str, int, List[int]],
):
    """Return saved‑figure path for a comparison plot with subplots for specific regions."""

    # --- Config & Style ------------------------------------------------------
    is_nsd = "nsd" in neural_dataset.lower()
    pca_sizes = [4, 8, 16, 32, 64] # PCA n_classes to plot, updated as per request
    target_regions_ordered = ["early visual stream", "midventral visual stream", "ventral visual stream"]

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'DejaVu Sans'], # Arial preferred, DejaVu Sans as fallback
        'font.size': 10,
        'axes.titlesize': 11,      # Subplot titles
        'axes.labelsize': 10,      # X and Y axis labels
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.titlesize': 13,    # Main figure suptitle
        'lines.linewidth': 1.75,
        'lines.markersize': 6,
        'axes.grid': True, # Ensure grid is on by default with this style
        'grid.linestyle': '--',
        'grid.alpha': 0.5
    })

    colours = {
        "initial": "#7f8c8d", # Grey
        "final": "#c0392b",   # Red
        "pca": dict(zip(pca_sizes, plt.cm.Blues(np.linspace(0.2, 0.8, len(pca_sizes))))), # Updated Blue hues
    }

    # --- Data Aggregation & Filtering ----------------------------------------
    def _col_present(df, col):
        return col in df.columns and (df[col] != "N/A").any()

    avg_subj = is_nsd and isinstance(subject_filter_val, list) and _col_present(full_df_filtered, "subject_idx")
    grp_reg = is_nsd and region_filter_val == "all" and _col_present(full_df_filtered, "region")

    base_cols = ["layer", "epoch"]
    if grp_reg:
        base_cols.insert(0, "region")
    
    pca_base_cols_for_agg = ["layer", "epoch", "pca_labels", "pca_n_classes"]
    if grp_reg:
        pca_base_cols_for_agg.insert(0, "region")

    def _agg(df: pd.DataFrame, cols_for_groupby):
        d = df.copy()

        current_avg_subj = avg_subj # Use the avg_subj determined outside _agg

        if current_avg_subj and "subject_idx" in d.columns:
            # Clean subject_idx: remove 'N/A', convert to numeric, drop resulting NaNs
            if pd.api.types.is_string_dtype(d["subject_idx"]):
                d = d[d["subject_idx"].astype(str).str.strip().lower() != "n/a"]
            
            # Convert to numeric, coercing errors. This handles existing numbers and cleaned strings.
            d["subject_idx"] = pd.to_numeric(d["subject_idx"], errors='coerce')
            d = d.dropna(subset=["subject_idx"]) # Remove rows where subject_idx became NaN

            if not d.empty:
                # First, average scores for each subject
                subject_mean_df = d.groupby(
                    cols_for_groupby + ["subject_idx"], observed=False
                )["score"].mean().reset_index()
                
                # Then, average these subject-specific means across subjects
                final_mean_df = subject_mean_df.groupby(
                    cols_for_groupby, observed=False
                )["score"].mean().reset_index()
                return final_mean_df
            else: # if d became empty after cleaning subject_idx
                # Return an empty DataFrame with expected columns if d is empty
                # This ensures consistency in what _agg returns.
                # Expected columns are cols_for_groupby + ['score']
                return pd.DataFrame(columns=cols_for_groupby + ['score'])

        # If not averaging subjects, or subject_idx not present, or d became empty before subject averaging step
        if d.empty: # Check again if previous operations (e.g. hypothetical other cleanups) emptied d
            return pd.DataFrame(columns=cols_for_groupby + ['score'])

        return d.groupby(cols_for_groupby, observed=False)["score"].mean().reset_index()

    agg_base_cols = base_cols[:] 
    agg_pca_cols = pca_base_cols_for_agg[:]

    full_agg = _agg(full_df_filtered, agg_base_cols)
    pca_agg  = _agg(pca_df_filtered, agg_pca_cols)

    if not grp_reg and region_filter_val != "all":
        if "region" not in full_agg.columns and not full_agg.empty: full_agg["region"] = region_filter_val
        if "region" not in pca_agg.columns and not pca_agg.empty: pca_agg["region"] = region_filter_val

    # Determine which of the target regions are actually present in the aggregated data
    all_unique_regions = set()
    if "region" in full_agg.columns and not full_agg.empty:
        all_unique_regions.update(full_agg["region"].dropna().unique())
    if "region" in pca_agg.columns and not pca_agg.empty:
        all_unique_regions.update(pca_agg["region"].dropna().unique())
    
    actual_regions_to_plot = [r for r in target_regions_ordered if r in all_unique_regions]

    if not actual_regions_to_plot:
        print(f"Warning: None of the target regions ({', '.join(target_regions_ordered)}) found in the provided data.")
        return []

    # --- Determine Global Y-axis Limits ---
    all_scores_for_ylim = []
    for region_name in actual_regions_to_plot:
        fm_region = full_agg[full_agg["region"] == region_name]
        pm_region = pca_agg[pca_agg["region"] == region_name]

        initial_region_scores = fm_region[fm_region["epoch"] == 0]["score"]
        if not initial_region_scores.empty: all_scores_for_ylim.append(initial_region_scores)
        
        final_region_scores = fm_region[fm_region["epoch"] == 18]["score"]
        if not final_region_scores.empty: all_scores_for_ylim.append(final_region_scores)

        for n_classes in pca_sizes:
            pca_n_region_scores = pm_region[(pm_region["pca_n_classes"] == n_classes) & (pm_region["epoch"] == 18)]["score"]
            if not pca_n_region_scores.empty: all_scores_for_ylim.append(pca_n_region_scores)
    
    min_y, max_y = None, None
    if all_scores_for_ylim:
        global_scores_series = pd.concat(all_scores_for_ylim).dropna()
        if not global_scores_series.empty:
            min_y = global_scores_series.min()
            max_y = global_scores_series.max()
            padding = (max_y - min_y) * 0.05 if (max_y - min_y) > 1e-9 else 0.05 # Avoid zero padding for flat lines
            min_y -= padding
            max_y += padding

    # --- Plotting ---
    num_subplots = len(actual_regions_to_plot)
    fig, axes = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 5.5), sharey=True)
    if num_subplots == 1: # Make axes iterable if only one subplot
        axes = [axes]

    x_ticks_pos = range(len(layer_order))

    for i, region_name in enumerate(actual_regions_to_plot):
        ax = axes[i]
        fm_subplot_data = full_agg[full_agg["region"] == region_name]
        pm_subplot_data = pca_agg[pca_agg["region"] == region_name]

        initial_data = fm_subplot_data[fm_subplot_data["epoch"] == 0].set_index("layer").reindex(layer_order)["score"]

        # Plot initial (full model, epoch 0)
        if not initial_data.empty and not initial_data.isna().all():
            ax.plot(x_ticks_pos, initial_data, color=colours["initial"], marker="x", linestyle="-", label="Initial", zorder=3)
        
        # Plot final (full model, epoch 18)
        final_data = fm_subplot_data[fm_subplot_data["epoch"] == 18].set_index("layer").reindex(layer_order)["score"]
        if not final_data.empty and not final_data.isna().all():
            ax.plot(x_ticks_pos, final_data, color=colours["final"], marker="o", linestyle="-", label="1000 Classes", zorder=3)

        # Plot PCA variants (epoch 18)
        pca_lines_data = {}
        for n_classes in pca_sizes:
            pca_lines_data[n_classes] = pm_subplot_data[
                (pm_subplot_data["pca_n_classes"] == n_classes) & (pm_subplot_data["epoch"] == 18)
            ].set_index("layer").reindex(layer_order)["score"]

        for n_classes, pca_series in pca_lines_data.items():
            if not pca_series.empty and not pca_series.isna().all():
                ax.plot(x_ticks_pos, pca_series, color=colours["pca"][n_classes], marker="s", linestyle="-", label=f"PCA {n_classes} classes", zorder=2)

        ax.set_xticks(x_ticks_pos)
        ax.set_xticklabels(layer_order, rotation=45, ha="right")
        # Make subplot titles slightly more compact if they contain "visual stream"
        clean_region_name = region_name.replace(" visual stream", "\nvisual stream").replace("midventral", "mid-ventral")
        ax.set_title(clean_region_name)

        if i == 0: # Y-label only for the first subplot
            ax.set_ylabel(f"RSA ({compare_rsm_correlation.capitalize()})")
        
        # ax.grid(True) # Grid is managed by style + rcParams
        
        if min_y is not None and max_y is not None: # Apply shared Y-limits
            ax.set_ylim(min_y, max_y)

    # --- Figure-level Adjustments ---
    fig_title_parts = [f"{neural_dataset.upper()} Comparison"]
    if isinstance(subject_filter_val, list) and len(subject_filter_val) < 8 : # Avoid overly long subject lists
         fig_title_parts.append(f"Subjects: {', '.join(map(str, subject_filter_val))}")
    elif isinstance(subject_filter_val, int) or (isinstance(subject_filter_val, str) and subject_filter_val != "all"):
         fig_title_parts.append(f"Subject: {subject_filter_val}")
    # fig_title_parts.append(f"Metric: {compare_rsm_correlation}") # Already in y-label
    # fig.suptitle(" - ".join(fig_title_parts)) # Removed suptitle as per request

    # Legend
    handles, labels = [], []
    for ax_check in axes: # Consolidate handles and labels from any axis that has them
        h_temp, l_temp = ax_check.get_legend_handles_labels()
        if h_temp:
            unique_labels_on_ax = {} # Deduplicate within this axis first
            for handle, label_text in zip(h_temp, l_temp):
                if label_text not in unique_labels_on_ax:
                    unique_labels_on_ax[label_text] = handle
            # Then add to global unique list
            for label_text, handle in unique_labels_on_ax.items():
                if label_text not in labels: # Maintain order of first appearance
                    labels.append(label_text)
                    handles.append(handle)
            # Break if we have representative set, assuming all subplots could have same set
            # For safety, let's collect from all and then make unique if needed, but current way should be fine.
            # If we want a specific order for legend items:
            # desired_label_order = ["Initial (Full Model, Epoch 0)", "Full Model (Epoch 18)"] + [f"PCA {n} classes (Epoch 18)" for n in pca_sizes]
            # ordered_handles = []
            # ordered_labels = []
            # temp_legend_dict = dict(zip(labels, handles))
            # for dl in desired_label_order:
            #     if dl in temp_legend_dict:
            #         ordered_labels.append(dl)
            #         ordered_handles.append(temp_legend_dict[dl])
            # handles, labels = ordered_handles, ordered_labels


    if handles: # Place legend below subplots
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.01), ncol=min(len(handles), 4), frameon=False)
        # Adjust layout to make space for legend and suptitle
        plt.tight_layout(rect=[0, 0.08, 1, 0.92]) # rect=[left, bottom, right, top]
    else:
        plt.tight_layout(rect=[0, 0.05, 1, 0.92])


    # --- Save Figure ---
    out_dir = "plotters/plots"
    os.makedirs(out_dir, exist_ok=True)
    
    subject_str_part = "allsubj"
    if isinstance(subject_filter_val, list) and len(subject_filter_val) < 8:
        subject_str_part = f"subj{'_'.join(map(str, subject_filter_val))}"
    elif isinstance(subject_filter_val, int) or (isinstance(subject_filter_val, str) and subject_filter_val != "all"):
        subject_str_part = f"subj{subject_filter_val}"

    regions_str_part = "_".join(s.replace(" visual stream", "").replace(" ", "") for s in actual_regions_to_plot)
    fname = f"comparison_{neural_dataset.lower()}_{regions_str_part}_{subject_str_part}_{compare_rsm_correlation.lower()}.png"
    
    path = os.path.join(out_dir, fname)
    plt.savefig(path, dpi=300, bbox_inches='tight') # Using bbox_inches='tight' again for safety
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
        "dataset": "nsd",  # 'nsd', 'nsd_synthetic', 'things', ...
        "metric": "Spearman",        # 'Pearson', 'Spearman', ...
        "region": "all",             # region name or 'all'
        "subject": [0, 1, 2, 3, 4, 5, 6, 7],      # int, list[int], or 'all'
        "layer_order": ["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2"],
        "log_dir": "logs", # Updated to be just logs/
        "combined_csv": "full_vs_pcs_nsd.csv", # New combined CSV
    }

    print("\n*** Plot generation cfg ***")
    for k, v in cfg.items():
        if k not in {"layer_order"}:
            print(f" {k}: {v}")

    # --- Load ----------------------------------------------------------------
    combined_raw = _load_csv(os.path.join(cfg["log_dir"], cfg["combined_csv"]))

    # Split into full_raw and pca_raw based on 'pca_labels' column
    # Ensure the column name matches exactly what's in your CSV.
    if "pca_labels" not in combined_raw.columns:
        raise ValueError("'pca_labels' column not found in the combined CSV file.")

    # Handle 'pca_labels' column which might be string "True"/"False" (case-insensitive, allowing whitespace)
    pca_labels_as_str_lower = combined_raw["pca_labels"].astype(str).str.strip().str.lower()
    full_raw = combined_raw[pca_labels_as_str_lower == "false"].copy()
    pca_raw  = combined_raw[pca_labels_as_str_lower == "true"].copy()


    # --- Apply original, verbose (working) filters ---------------------------
    ds = cfg["dataset"].lower().strip() # Ensure ds from cfg is also clean
    metric = cfg["metric"].strip()      # Ensure metric from cfg is also clean
    region_filter_key = cfg["region"].strip() # Ensure region from cfg is also clean

    print("\nApplying filters…")

    # Dataset filter (case-insensitive, strip whitespace)
    # Ensure "neural_dataset" column exists and handle potential errors if it's not string type
    if "neural_dataset" in full_raw.columns:
        full = full_raw[full_raw["neural_dataset"].astype(str).str.strip().str.lower() == ds].copy()
    else:
        print("  Warning: 'neural_dataset' column missing in full_raw. It will be empty.")
        full = pd.DataFrame(columns=full_raw.columns) # Create empty DataFrame with same columns

    if "neural_dataset" in pca_raw.columns:
        pca  = pca_raw[pca_raw["neural_dataset"].astype(str).str.strip().str.lower() == ds].copy()
    else:
        print("  Warning: 'neural_dataset' column missing in pca_raw. It will be empty.")
        pca = pd.DataFrame(columns=pca_raw.columns)   # Create empty DataFrame with same columns

    print(f"  neural_dataset='{ds}': {len(full)} (full), {len(pca)} (pca)")

    # Metric filter
    for df, name in ((full, "full"), (pca, "pca")):
        if "compare_rsm_correlation" in df.columns:
            # Ensure a consistent comparison by stripping and lowercasing if metric is a string type
            # However, metric from cfg is already stripped.
            # The column itself should be robustly compared
            condition = df["compare_rsm_correlation"].astype(str).str.strip() != metric
            df.drop(df[condition].index, inplace=True)
        else:
            print(f"  Warning: 'compare_rsm_correlation' missing in {name} CSV")
    print(f"  metric='{metric}': {len(full)} (full), {len(pca)} (pca)")

    # Region filter
    if region_filter_key != "all":
        for df, name in ((full, "full"), (pca, "pca")):
            if "region" in df.columns:
                condition = df["region"].astype(str).str.strip() != region_filter_key
                df.drop(df[condition].index, inplace=True)
            else:
                print(f"  Warning: 'region' missing in {name} CSV; emptied")
                df.drop(df.index, inplace=True)
        print(f"  region='{region_filter_key}': {len(full)} (full), {len(pca)} (pca)")
    else:
        print("  region='all' (no filter)")

    # Subject filter
    if cfg["subject"] != "all":
        subj_list = cfg["subject"] if isinstance(cfg["subject"], list) else [cfg["subject"]]
        subj_list = [int(s) for s in subj_list]
        for df, name in ((full, "full"), (pca, "pca")):
            if "subject_idx" in df.columns:
                df["subject_idx_num"] = pd.to_numeric(df["subject_idx"], errors="coerce")
                df.drop(df[~df["subject_idx_num"].isin(subj_list)].index, inplace=True)
                df.drop(columns=["subject_idx_num"], inplace=True)
            else:
                print(f"  Warning: 'subject_idx' missing in {name} CSV; emptied")
                df.drop(df.index, inplace=True)
        print(f"  subject={subj_list}: {len(full)} (full), {len(pca)} (pca)")
    else:
        print("  subject='all' (no filter)")

    # --- Validate ------------------------------------------------------------
    if full.empty or pca.empty:
        raise ValueError("Empty dataframe(s) after filtering. Check filters and 'pca_labels' column in CSV.")

    # --- Plot ----------------------------------------------------------------
    paths = create_comparison_plots(
        full,
        pca,
        cfg["layer_order"],
        cfg["dataset"],
        cfg["metric"],
        cfg["region"],
        cfg["subject"],
    )

    print(f"\nGenerated {len(paths)} plot(s):")
    for p in paths:
        print("  •", p)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("Error:", exc)
