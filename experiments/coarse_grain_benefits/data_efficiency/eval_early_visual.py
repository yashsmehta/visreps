"""
Evaluate data-efficiency models on NSD early visual stream at the same
epoch selected as best for ventral visual stream (from existing CSV results).

Saves results to data_efficiency_results.csv with benchmark="nsd_early".

Usage (from project root):
    python experiments/coarse_grain_benefits/data_efficiency/eval_early_visual.py
    python experiments/coarse_grain_benefits/data_efficiency/eval_early_visual.py --pca_labels alexnet
"""

import gc
import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

import torch
import pandas as pd
import visreps.evals as evals
from visreps.utils import load_config, validate_config
from experiments.coarse_grain_benefits.data_efficiency.shared import (
    SEED, DEFAULT_PCA_LABELS, DATASETS,
    get_conditions, get_checkpoint_dir, get_csv_path, save_results,
)


def get_best_ventral_epochs(csv_path, conditions):
    """Read existing CSV and find best ventral epoch per (dataset, condition)."""
    df = pd.read_csv(csv_path)
    nsd = df[df["benchmark"] == "nsd"]

    best_epochs = {}
    for ds in DATASETS:
        dsdf = nsd[nsd["dataset"] == ds]
        for cond in conditions:
            cdf = dsdf[dsdf["condition"] == cond]
            if cdf.empty:
                continue
            # Average across subjects per epoch, pick epoch with highest mean
            epoch_means = cdf.groupby("epoch")["score"].mean()
            best_epoch = int(epoch_means.idxmax())
            best_epochs[(ds, cond)] = best_epoch
    return best_epochs


def eval_early_visual(dataset, condition_id, epoch, conditions, pca_labels):
    """Run NSD early visual stream eval, return per-subject result rows."""
    print(f"\n{'='*60}")
    print(f"Early visual stream: {condition_id}-class | {dataset} | epoch {epoch}")
    print(f"{'='*60}")

    cond = conditions[condition_id]
    checkpoint_dir = f"model_checkpoints/{get_checkpoint_dir(dataset, pca_labels)}"

    overrides = [
        f"seed={SEED}",
        f"cfg_id={condition_id}",
        f"checkpoint_dir={checkpoint_dir}",
        f"checkpoint_model=checkpoint_epoch_{epoch}.pth",
        f"pca_labels={cond['pca_labels']}",
        f"pca_n_classes={cond['pca_n_classes']}",
        "analysis=rsa",
        "compare_method=spearman",
        "bootstrap=true",
        "load_model_from=checkpoint",
        "log_expdata=false",
        "batchsize=256",
        "num_workers=0",
        "neural_dataset=nsd",
        "region=early visual stream",
        "subject_idx=[0,1,2,3,4,5,6,7]",
    ]
    if "pca_labels_folder" in cond:
        overrides.append(f"pca_labels_folder={cond['pca_labels_folder']}")

    cfg = load_config("configs/eval/base.json", overrides)
    cfg = validate_config(cfg)
    result_df = evals.eval(cfg)

    rows = []
    for i, (_, r) in enumerate(result_df.iterrows()):
        rows.append({
            "dataset": dataset,
            "condition": condition_id,
            "epoch": epoch,
            "benchmark": "nsd_early",
            "subject_idx": i,
            "layer": r["layer"],
            "score": round(r["score"], 4),
            "ci_low": round(r["ci_low"], 4) if r.get("ci_low") is not None else None,
            "ci_high": round(r["ci_high"], 4) if r.get("ci_high") is not None else None,
        })
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate data efficiency models on NSD early visual stream")
    parser.add_argument("--pca_labels", type=str, default=DEFAULT_PCA_LABELS,
                        help="PCA labels source, e.g. 'clip' or 'alexnet' (default: clip)")
    parser.add_argument("--force", action="store_true",
                        help="Re-evaluate even if result exists in CSV")
    args = parser.parse_args()

    conditions = get_conditions(args.pca_labels)
    csv_path = get_csv_path(args.pca_labels)
    best_epochs = get_best_ventral_epochs(csv_path, conditions)

    print("Best ventral epochs (will evaluate early visual stream at these):")
    for (ds, cond), epoch in sorted(best_epochs.items()):
        print(f"  {ds} / {cond}-class -> epoch {epoch}")

    # Check which are already done
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        existing = set(zip(df["dataset"], df["condition"], df["epoch"], df["benchmark"]))
    else:
        existing = set()

    completed, skipped = 0, 0
    for (ds, cond), epoch in sorted(best_epochs.items()):
        if not args.force and (ds, cond, epoch, "nsd_early") in existing:
            print(f"[SKIP] {cond}-class {ds} epoch {epoch} — already evaluated")
            skipped += 1
            continue

        rows = eval_early_visual(ds, cond, epoch, conditions, args.pca_labels)
        save_results(rows, csv_path)
        completed += 1

        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nDone. {completed} evals run, {skipped} skipped.")


if __name__ == "__main__":
    main()
