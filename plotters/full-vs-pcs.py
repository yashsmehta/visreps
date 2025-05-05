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
    """Return saved‑figure paths comparing the full model to PCA variants."""

    is_nsd = "nsd" in neural_dataset.lower()
    pca_sizes = [2, 4, 8, 16, 32, 64]

    # --- Colours -------------------------------------------------------------
    colours = {
        "initial": "#7f8c8d",
        "final": "#c0392b",
        "pca": dict(zip(pca_sizes, plt.cm.Blues(np.linspace(0.3, 0.9, len(pca_sizes))))),
    }

    # --- Decide aggregation --------------------------------------------------
    def _col_present(df, col):
        return col in df.columns and (df[col] != "N/A").any()

    avg_subj = is_nsd and isinstance(subject_filter_val, list) and _col_present(full_df_filtered, "subject_idx")
    grp_reg  = is_nsd and region_filter_val == "all" and _col_present(full_df_filtered, "region")

    base_cols = ["layer", "epoch"]
    pca_cols  = base_cols + ["pca_labels", "pca_n_classes"]
    if grp_reg:
        base_cols.insert(0, "region")
        pca_cols.insert(0, "region")

    # --- Aggregation helper --------------------------------------------------
    def _agg(df: pd.DataFrame, cols):
        d = df.copy()
        if avg_subj and "subject_idx" in d.columns:
            d = d[d["subject_idx"] != "N/A"]
        if grp_reg and "region" in d.columns:
            d = d[d["region"] != "N/A"]

        if avg_subj and "subject_idx" in d.columns:
            d = (
                d.groupby(cols + ["subject_idx"], observed=False)["score"].mean().reset_index()
            )
        return d.groupby(cols, observed=False)["score"].mean().reset_index()

    full = _agg(full_df_filtered, base_cols)
    pca  = _agg(pca_df_filtered, pca_cols)

    if not grp_reg:
        reg_val = region_filter_val if region_filter_val != "all" else "all"
        full["region"] = reg_val
        pca["region"]  = reg_val

    regions = sorted(full["region"].unique())

    # --- Plotting ------------------------------------------------------------
    out_dir = "plotters/plots"
    os.makedirs(out_dir, exist_ok=True)
    paths = []

    for region in regions:
        fm = full[full["region"] == region]
        pm = pca[pca["region"] == region]

        initial = fm[fm["epoch"] == 0].set_index("layer").reindex(layer_order)
        final   = fm[fm["epoch"] == 10].set_index("layer").reindex(layer_order)

        pca_lines = {
            n: pm[(pm["pca_n_classes"] == n) & (pm["epoch"] == 10)]
            .set_index("layer").reindex(layer_order)
            for n in pca_sizes
        }

        if initial.empty or final.empty or all(df.empty for df in pca_lines.values()):
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        x = range(len(layer_order))
        ax.plot(x, initial["score"], color=colours["initial"], marker="x", label="Initial")
        ax.plot(x, final["score"],   color=colours["final"],   marker="o", label="Full")

        for n, df in pca_lines.items():
            if df["score"].notna().any():
                ax.plot(x, df["score"], color=colours["pca"][n], marker="s", label=str(n))

        ax.set_xticks(x)
        ax.set_xticklabels(layer_order, rotation=45, ha="right")
        ax.set_ylabel(f"{compare_rsm_correlation.capitalize()} score")

        title = f"{neural_dataset.upper()} – {region}"
        if isinstance(subject_filter_val, list):
            title += f" subjects {subject_filter_val}"
        elif subject_filter_val != "all":
            title += f" sub {subject_filter_val}"
        ax.set_title(title)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout(rect=[0, 0, 0.8, 1])

        fname = f"full_vs_pcs_{neural_dataset}_{region}_{compare_rsm_correlation}.png".replace(" ", "_")
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
    pca_raw  = _load_csv(os.path.join(cfg["log_dir"], cfg["pca_csv"]))

    # --- Apply original, verbose (working) filters ---------------------------
    ds = cfg["dataset"].lower()
    metric = cfg["metric"]
    region = cfg["region"]
    subj_filt = cfg["subject"]

    print("\nApplying filters…")

    # Dataset filter (case‑insensitive)
    full = full_raw[full_raw["neural_dataset"].str.lower() == ds].copy()
    pca  = pca_raw[pca_raw["neural_dataset"].str.lower() == ds].copy()
    print(f"  neural_dataset='{ds}': {len(full)} (full), {len(pca)} (pca)")

    # Metric filter
    for df, name in ((full, "full"), (pca, "pca")):
        if "compare_rsm_correlation" in df.columns:
            df.drop(df[df["compare_rsm_correlation"] != metric].index, inplace=True)
        else:
            print(f"  Warning: 'compare_rsm_correlation' missing in {name} CSV")
    print(f"  metric='{metric}': {len(full)} (full), {len(pca)} (pca)")

    # Region filter
    if region != "all":
        for df, name in ((full, "full"), (pca, "pca")):
            if "region" in df.columns:
                df.drop(df[df["region"] != region].index, inplace=True)
            else:
                print(f"  Warning: 'region' missing in {name} CSV; emptied")
                df.drop(df.index, inplace=True)
        print(f"  region='{region}': {len(full)} (full), {len(pca)} (pca)")
    else:
        print("  region='all' (no filter)")

    # Subject filter
    if subj_filt != "all":
        subj_list = subj_filt if isinstance(subj_filt, list) else [subj_filt]
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
        raise ValueError("Empty dataframe(s) after filtering.")

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
