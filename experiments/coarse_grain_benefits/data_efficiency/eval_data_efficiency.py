"""
Evaluate data-efficiency trained models on THINGS behavioral alignment,
NSD ventral visual stream, and TVSD IT. Uses the standard two-phase RSA
pipeline (layer selection on train, re-extract best layer on test).

Results are saved to a single combined CSV: data_efficiency_results.csv
(or data_efficiency_{pca_labels}_results.csv for non-default PCA labels).

Usage (from project root):
    python experiments/coarse_grain_benefits/data_efficiency/eval_data_efficiency.py
    python experiments/coarse_grain_benefits/data_efficiency/eval_data_efficiency.py --pca_labels alexnet
    python experiments/coarse_grain_benefits/data_efficiency/eval_data_efficiency.py --datasets imagenet-mini-50
    python experiments/coarse_grain_benefits/data_efficiency/eval_data_efficiency.py --conditions 16 32
    python experiments/coarse_grain_benefits/data_efficiency/eval_data_efficiency.py --benchmarks things
    python experiments/coarse_grain_benefits/data_efficiency/eval_data_efficiency.py --benchmarks tvsd
    python experiments/coarse_grain_benefits/data_efficiency/eval_data_efficiency.py --print_only
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
    SEED, DEFAULT_PCA_LABELS, DATASETS, EPOCHS,
    get_conditions, get_checkpoint_dir, get_csv_path, save_results,
)


def build_overrides(dataset, condition_id, epoch, benchmark, conditions, pca_labels):
    """Build config overrides for a single eval run."""
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
    ]
    if "pca_labels_folder" in cond:
        overrides.append(f"pca_labels_folder={cond['pca_labels_folder']}")

    if benchmark == "things":
        overrides.append("neural_dataset=things-behavior")
    elif benchmark == "nsd":
        overrides.append("neural_dataset=nsd")
        overrides.append("region=ventral visual stream")
        overrides.append("subject_idx=[0,1,2,3,4,5,6,7]")
    elif benchmark == "tvsd":
        overrides.append("neural_dataset=tvsd")
        overrides.append("region=IT")
        overrides.append("subject_idx=[0,1]")

    return overrides


def load_existing_results(csv_path):
    """Load existing CSV once and return set of completed (dataset, condition, epoch, benchmark) tuples."""
    if not os.path.exists(csv_path):
        return set()
    df = pd.read_csv(csv_path)
    return set(zip(df["dataset"], df["condition"], df["epoch"], df["benchmark"]))


def result_exists(completed, dataset, condition_id, epoch, benchmark):
    """Check if result already exists using pre-loaded set."""
    return (dataset, condition_id, epoch, benchmark) in completed


def checkpoint_exists(dataset, condition_id, epoch, pca_labels):
    """Check if the checkpoint file exists."""
    checkpoint_dir = get_checkpoint_dir(dataset, pca_labels)
    seed_letter = "a"  # seed=1
    path = os.path.join("model_checkpoints", checkpoint_dir,
                        f"cfg{condition_id}{seed_letter}",
                        f"checkpoint_epoch_{epoch}.pth")
    return os.path.exists(path)


def _make_result_rows(result_df, dataset, condition_id, epoch, benchmark):
    """Convert eval result DataFrame to list of row dicts for the CSV."""
    rows = []
    for i, (_, r) in enumerate(result_df.iterrows()):
        rows.append({
            "dataset": dataset,
            "condition": condition_id,
            "epoch": epoch,
            "benchmark": benchmark,
            "subject_idx": "N/A" if benchmark == "things" else i,
            "layer": r["layer"],
            "score": round(r["score"], 4),
            "ci_low": round(r["ci_low"], 4) if r.get("ci_low") is not None else None,
            "ci_high": round(r["ci_high"], 4) if r.get("ci_high") is not None else None,
        })
    return rows


def eval_benchmark(dataset, condition_id, epoch, benchmark, conditions, pca_labels):
    """Run a single evaluation and return result rows."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {condition_id}-class | {dataset} | epoch {epoch} | {benchmark}")
    print(f"{'='*60}")

    overrides = build_overrides(dataset, condition_id, epoch, benchmark, conditions, pca_labels)
    cfg = load_config("configs/eval/base.json", overrides)
    cfg = validate_config(cfg)
    result_df = evals.eval(cfg)
    return _make_result_rows(result_df, dataset, condition_id, epoch, benchmark)


def print_summary(datasets, conditions, benchmarks, csv_path):
    """Print results summary table."""
    if not os.path.exists(csv_path):
        print("No results CSV found.")
        return

    df = pd.read_csv(csv_path)

    for bench in benchmarks:
        bdf = df[df["benchmark"] == bench]
        if len(bdf) == 0:
            continue

        print(f"\n{'='*70}")
        if bench == "things":
            print("THINGS Behavioral Alignment — Data Efficiency")
        elif bench == "nsd":
            print("NSD Ventral Stream Alignment — Data Efficiency (mean across subjects)")
        elif bench == "tvsd":
            print("TVSD IT Alignment — Data Efficiency (mean across subjects)")
        print(f"{'='*70}")
        print(f"{'Dataset':<22} {'Condition':>10} {'Epoch':>6} {'Score':>8}")
        print(f"{'-'*70}")

        for ds in datasets:
            ds_df = bdf[bdf["dataset"] == ds]
            if len(ds_df) == 0:
                continue
            for cond in conditions:
                cond_df = ds_df[ds_df["condition"] == cond]
                for epoch in EPOCHS:
                    edf = cond_df[cond_df["epoch"] == epoch]
                    if len(edf) == 0:
                        continue
                    score = edf["score"].mean()
                    print(f"{ds:<22} {cond:>10} {epoch:>6} {score:>8.4f}")

        print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate data efficiency models on THINGS, NSD, and TVSD")
    parser.add_argument("--pca_labels", type=str, default=DEFAULT_PCA_LABELS,
                        help="PCA labels source, e.g. 'clip' or 'alexnet' (default: clip)")
    parser.add_argument("--datasets", type=str, nargs="+", default=DATASETS,
                        choices=DATASETS)
    parser.add_argument("--conditions", type=int, nargs="+", default=None,
                        help="Which conditions to evaluate (default: all)")
    parser.add_argument("--epochs", type=int, nargs="+", default=EPOCHS)
    parser.add_argument("--benchmarks", type=str, nargs="+",
                        default=["things", "nsd", "tvsd"],
                        choices=["things", "nsd", "tvsd"])
    parser.add_argument("--force", action="store_true",
                        help="Re-evaluate even if result exists in CSV")
    parser.add_argument("--no_bootstrap", action="store_true")
    parser.add_argument("--print_only", action="store_true")
    args = parser.parse_args()

    conditions = get_conditions(args.pca_labels)
    if args.conditions is None:
        args.conditions = list(conditions.keys())
    csv_path = get_csv_path(args.pca_labels)

    if args.print_only:
        print_summary(args.datasets, args.conditions, args.benchmarks, csv_path)
        return

    total = len(args.datasets) * len(args.conditions) * len(args.epochs) * len(args.benchmarks)
    completed, skipped = 0, 0
    existing = load_existing_results(csv_path)

    for dataset in args.datasets:
        for condition_id in args.conditions:
            for epoch in args.epochs:
                for benchmark in args.benchmarks:
                    # Skip if checkpoint doesn't exist
                    if not checkpoint_exists(dataset, condition_id, epoch, args.pca_labels):
                        print(f"[SKIP] {condition_id}-class {dataset} epoch {epoch} "
                              f"— checkpoint not found")
                        skipped += 1
                        continue

                    # Skip if result already in CSV
                    if not args.force and result_exists(existing, dataset, condition_id, epoch, benchmark):
                        print(f"[SKIP] {condition_id}-class {dataset} epoch {epoch} "
                              f"{benchmark} — already evaluated")
                        skipped += 1
                        continue

                    rows = eval_benchmark(dataset, condition_id, epoch, benchmark,
                                          conditions, args.pca_labels)
                    save_results(rows, csv_path)
                    completed += 1

                    # Release file handles between runs to avoid "Too many open files"
                    gc.collect()
                    torch.cuda.empty_cache()

    print(f"\nEvaluation complete. {completed}/{total} runs executed, {skipped} skipped.")
    print_summary(args.datasets, args.conditions, args.benchmarks, csv_path)


if __name__ == "__main__":
    main()
