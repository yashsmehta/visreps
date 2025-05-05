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
    layer_order: List[str],
    neural_dataset: str,
    compare_rsm_correlation: str,
    region_filter_val: str,
    subject_filter_val: Union[str, int, List[int]],
):
    """Return saved‑figure paths showing variance in the full model across subjects."""

    is_nsd = "nsd" in neural_dataset.lower()

    # --- Colours -------------------------------------------------------------
    colours = {
        "initial": "#7f8c8d",
        "final": "#c0392b",
        "initial_box": "#bdc3c7",
        "final_box": "#e74c3c",
    }

    # --- Decide aggregation --------------------------------------------------
    def _col_present(df, col):
        return col in df.columns and (df[col] != "N/A").any()

    avg_subj = is_nsd and isinstance(subject_filter_val, list) and _col_present(full_df_filtered, "subject_idx")
    grp_reg  = is_nsd and region_filter_val == "all" and _col_present(full_df_filtered, "region")

    base_cols = ["layer", "epoch"]
    if grp_reg:
        base_cols.insert(0, "region")

    # --- Aggregation helper for means -----------------------------------------
    def _agg_mean(df: pd.DataFrame, cols):
        d = df.copy()
        group_cols = cols
        if avg_subj and "subject_idx" in d.columns:
            d = d[d["subject_idx"] != "N/A"]
            group_cols = cols + ["subject_idx"]

        if grp_reg and "region" in d.columns:
            d = d[d["region"] != "N/A"]

        if avg_subj and "subject_idx" in d.columns:
            d_mean_subj = d.groupby(group_cols, observed=False)["score"].mean().reset_index()
            return d_mean_subj.groupby(cols, observed=False)["score"].mean().reset_index()
        else:
            return d.groupby(cols, observed=False)["score"].mean().reset_index()

    full_mean = _agg_mean(full_df_filtered, base_cols)

    # Prepare data for box plots (keep individual subject scores)
    full_box_data = full_df_filtered.copy()
    if avg_subj and "subject_idx" in full_box_data.columns:
        full_box_data = full_box_data[full_box_data["subject_idx"] != "N/A"]
    if grp_reg and "region" in full_box_data.columns:
        full_box_data = full_box_data[full_box_data["region"] != "N/A"]

    if not grp_reg:
        reg_val = region_filter_val if region_filter_val != "all" else "all"
        full_mean["region"] = reg_val
        full_box_data["region"] = reg_val

    regions = sorted(full_mean["region"].unique())

    # --- Plotting ------------------------------------------------------------
    out_dir = "plotters/plots"
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    for region in regions:
        fm = full_mean[full_mean["region"] == region]
        fbd = full_box_data[full_box_data["region"] == region]

        initial_mean = fm[fm["epoch"] == 0].set_index("layer").reindex(layer_order)["score"]
        final_mean   = fm[fm["epoch"] == 10].set_index("layer").reindex(layer_order)["score"]

        initial_box = [
            fbd[(fbd["epoch"] == 0) & (fbd["layer"] == layer)]["score"].dropna().tolist()
            for layer in layer_order
        ]
        final_box = [
            fbd[(fbd["epoch"] == 10) & (fbd["layer"] == layer)]["score"].dropna().tolist()
            for layer in layer_order
        ]

        has_initial_mean = not initial_mean.isna().all()
        has_final_mean = not final_mean.isna().all()
        has_initial_box = any(initial_box)
        has_final_box = any(final_box)

        if not (has_initial_mean or has_final_mean or has_initial_box or has_final_box):
            print(f"Skipping region '{region}' due to insufficient data.")
            continue

        fig, ax = plt.subplots(figsize=(12, 7))
        x = np.arange(len(layer_order))
        box_width = 0.2

        if has_initial_mean:
            ax.plot(x, initial_mean, color=colours["initial"], marker="x", linestyle="-", linewidth=3.0, label="Initial Mean")
        if has_final_mean:
            ax.plot(x, final_mean,   color=colours["final"],   marker="o", linestyle="-", linewidth=3.0, label="Full Mean")

        if has_initial_box:
            bp_initial = ax.boxplot(
                initial_box, positions=x - box_width/2, widths=box_width,
                patch_artist=True, showfliers=False,
                boxprops=dict(facecolor=colours["initial_box"], alpha=0.3),
                medianprops=dict(color=colours["initial"])
            )
        if has_final_box:
            bp_final = ax.boxplot(
                final_box, positions=x + box_width/2, widths=box_width,
                patch_artist=True, showfliers=False,
                boxprops=dict(facecolor=colours["final_box"], alpha=0.3),
                medianprops=dict(color=colours["final"])
            )

        if has_initial_box:
            ax.plot([], [], color=colours["initial_box"], linewidth=10, alpha=0.3, label='Initial Dist.')
        if has_final_box:
            ax.plot([], [], color=colours["final_box"], linewidth=10, alpha=0.3, label='Full Dist.')

        ax.set_xticks(x)
        ax.set_xticklabels(layer_order, rotation=45, ha="right")
        ax.set_ylabel(f"{compare_rsm_correlation.capitalize()} score")

        title = f"{neural_dataset.upper()} – {region}"
        if isinstance(subject_filter_val, list):
            title += f" subjects {subject_filter_val} (N={len(subject_filter_val)})"
        elif subject_filter_val != "all":
            title += f" sub {subject_filter_val}"
        else:
            if avg_subj and "subject_idx" in full_box_data.columns:
                n_subs = full_box_data['subject_idx'].nunique()
                title += f" (all subjects, N={n_subs})"

        ax.set_title(title)
        ax.grid(True, axis='y', linestyle="--", alpha=0.3)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout(rect=[0, 0, 0.85, 1])

        fname = f"full_variance_{neural_dataset}_{region}_{compare_rsm_correlation}.png".replace(" ", "_")
        path = os.path.join(out_dir, fname)
        plt.savefig(path, dpi=300)
        plt.close(fig)
        paths.append(path)

    return paths

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
        "dataset": "nsd_synthetic",  # 'nsd', 'nsd_synthetic', 'things', ...
        "metric": "Spearman",        # 'Pearson', 'Spearman', ...
        "region": "all",             # region name or 'all'
        "subject": [0, 1, 2, 3, 4, 5, 6, 7],      # int, list[int], or 'all'
        "layer_order": ["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2"],
        "log_dir": "logs/eval/checkpoint",
        "full_csv": "imagenet_cnn.csv",
        "pca_csv": "imagenet_pca.csv",
    }

    print("\n*** Plot generation cfg ***")
    for k, v in cfg.items():
        if k not in {"layer_order"}:
            print(f" {k}: {v}")

    # --- Load ----------------------------------------------------------------
    full_raw = _load_csv(os.path.join(cfg["log_dir"], cfg["full_csv"]))

    # --- Apply original, verbose (working) filters ---------------------------
    ds = cfg["dataset"].lower()
    metric = cfg["metric"]
    region = cfg["region"]
    subj_filt = cfg["subject"]

    print("\nApplying filters…")

    # Dataset filter (case‑insensitive)
    full = full_raw[full_raw["neural_dataset"].str.lower() == ds].copy()
    print(f"  neural_dataset='{ds}': {len(full)} rows")

    # Metric filter
    if "compare_rsm_correlation" in full.columns:
        full.drop(full[full["compare_rsm_correlation"] != metric].index, inplace=True)
    else:
        print(f"  Warning: 'compare_rsm_correlation' missing in full CSV")
    print(f"  metric='{metric}': {len(full)} rows")

    # Region filter
    if region != "all":
        if "region" in full.columns:
            full.drop(full[full["region"] != region].index, inplace=True)
        else:
            print(f"  Warning: 'region' missing in full CSV; emptied")
            full.drop(full.index, inplace=True)
        print(f"  region='{region}': {len(full)} rows")
    else:
        print("  region='all' (no filter)")

    # Subject filter
    if subj_filt != "all":
        subj_list = subj_filt if isinstance(subj_filt, list) else [subj_filt]
        subj_list = [int(s) for s in subj_list]
        if "subject_idx" in full.columns:
            full["subject_idx_num"] = pd.to_numeric(full["subject_idx"], errors="coerce")
            full.drop(full[~full["subject_idx_num"].isin(subj_list)].index, inplace=True)
            full.drop(columns=["subject_idx_num"], inplace=True)
        else:
            print(f"  Warning: 'subject_idx' missing in full CSV; emptied")
            full.drop(full.index, inplace=True)
        print(f"  subject={subj_list}: {len(full)} rows")
    else:
        print("  subject='all' (no filter)")

    # --- Validate ------------------------------------------------------------
    if full.empty:
        raise ValueError("Full dataframe empty after filtering.")

    # --- Plot ----------------------------------------------------------------
    paths = create_comparison_plots(
        full,
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
