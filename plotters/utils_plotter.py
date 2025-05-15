from typing import List, Optional
import pandas as pd

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
        • pca_df  – rows with pca_labels == True
        • full_df – rows with pca_labels == False

    Each optional argument filters rows if provided (None → no filter).
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