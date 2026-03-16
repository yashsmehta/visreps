"""
Evaluate data-efficiency trained models on THINGS behavioral alignment,
NSD ventral visual stream, and TVSD IT. Uses the standard two-phase RSA
pipeline (layer selection on train, re-extract best layer on test).

Results are saved to a single combined CSV: data_efficiency_results.csv

Usage (from project root):
    python experiments/coarse_grain_benefits/data_efficiency/eval_data_efficiency.py
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

SEED = 1
DATASETS = ["imagenet-mini-5", "imagenet-mini-10", "imagenet-mini-50"]
CONDITIONS = {
    8:    {"pca_labels": True, "pca_n_classes": 8,  "pca_labels_folder": "pca_labels_clip"},
    16:   {"pca_labels": True, "pca_n_classes": 16, "pca_labels_folder": "pca_labels_clip"},
    32:   {"pca_labels": True, "pca_n_classes": 32, "pca_labels_folder": "pca_labels_clip"},
    64:   {"pca_labels": True, "pca_n_classes": 64, "pca_labels_folder": "pca_labels_clip"},
    1000: {"pca_labels": False, "pca_n_classes": 1000},
}
EPOCHS = [100, 200]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(SCRIPT_DIR, "data_efficiency_results.csv")


def build_overrides(dataset, condition_id, epoch, benchmark):
    """Build config overrides for a single eval run."""
    cond = CONDITIONS[condition_id]
    checkpoint_dir = f"model_checkpoints/data_efficiency_{dataset}"

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


def load_existing_results():
    """Load existing CSV once and return set of completed (dataset, condition, epoch, benchmark) tuples."""
    if not os.path.exists(CSV_PATH):
        return set()
    df = pd.read_csv(CSV_PATH)
    return set(zip(df["dataset"], df["condition"], df["epoch"], df["benchmark"]))


def result_exists(completed, dataset, condition_id, epoch, benchmark):
    """Check if result already exists using pre-loaded set."""
    return (dataset, condition_id, epoch, benchmark) in completed


def checkpoint_exists(dataset, condition_id, epoch):
    """Check if the checkpoint file exists."""
    checkpoint_dir = f"model_checkpoints/data_efficiency_{dataset}"
    seed_letter = "a"  # seed=1
    path = os.path.join(checkpoint_dir, f"cfg{condition_id}{seed_letter}",
                        f"checkpoint_epoch_{epoch}.pth")
    return os.path.exists(path)


def save_results(rows):
    """Append result rows to the combined CSV, deduplicating."""
    new_df = pd.DataFrame(rows)
    if os.path.exists(CSV_PATH):
        existing = pd.read_csv(CSV_PATH)
        for _, row in new_df.iterrows():
            mask = (
                (existing["dataset"] == row["dataset"]) &
                (existing["condition"] == row["condition"]) &
                (existing["epoch"] == row["epoch"]) &
                (existing["benchmark"] == row["benchmark"]) &
                (existing["subject_idx"] == row["subject_idx"])
            )
            existing = existing[~mask]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined = combined.sort_values(
        ["benchmark", "dataset", "condition", "epoch", "subject_idx"]
    ).reset_index(drop=True)
    combined.to_csv(CSV_PATH, index=False)
    print(f"  Saved to {CSV_PATH}")


def eval_run(dataset, condition_id, epoch, benchmark):
    """Run a single evaluation and return result rows."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {condition_id}-class | {dataset} | epoch {epoch} | {benchmark}")
    print(f"{'='*60}")

    overrides = build_overrides(dataset, condition_id, epoch, benchmark)
    cfg = load_config("configs/eval/base.json", overrides)
    cfg = validate_config(cfg)
    result_df = evals.eval(cfg)

    rows = []
    for _, r in result_df.iterrows():
        row = {
            "dataset": dataset,
            "condition": condition_id,
            "epoch": epoch,
            "benchmark": benchmark,
            "subject_idx": "N/A" if benchmark == "things" else r.get("subject_idx", "N/A"),
            "layer": r["layer"],
            "score": round(r["score"], 4),
            "ci_low": round(r["ci_low"], 4) if r.get("ci_low") is not None else None,
            "ci_high": round(r["ci_high"], 4) if r.get("ci_high") is not None else None,
        }
        rows.append(row)

    return rows


def eval_nsd(dataset, condition_id, epoch):
    """Run NSD eval — returns per-subject rows with subject_idx populated."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {condition_id}-class | {dataset} | epoch {epoch} | nsd")
    print(f"{'='*60}")

    overrides = build_overrides(dataset, condition_id, epoch, "nsd")
    cfg = load_config("configs/eval/base.json", overrides)
    cfg = validate_config(cfg)
    result_df = evals.eval(cfg)

    # The eval returns one row per (subject, region) pair.
    # We need to figure out subject_idx from the order (0-7).
    rows = []
    for i, (_, r) in enumerate(result_df.iterrows()):
        rows.append({
            "dataset": dataset,
            "condition": condition_id,
            "epoch": epoch,
            "benchmark": "nsd",
            "subject_idx": i,
            "layer": r["layer"],
            "score": round(r["score"], 4),
            "ci_low": round(r["ci_low"], 4) if r.get("ci_low") is not None else None,
            "ci_high": round(r["ci_high"], 4) if r.get("ci_high") is not None else None,
        })

    return rows


def eval_tvsd(dataset, condition_id, epoch):
    """Run TVSD IT eval — returns per-subject rows (2 monkeys)."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {condition_id}-class | {dataset} | epoch {epoch} | tvsd")
    print(f"{'='*60}")

    overrides = build_overrides(dataset, condition_id, epoch, "tvsd")
    cfg = load_config("configs/eval/base.json", overrides)
    cfg = validate_config(cfg)
    result_df = evals.eval(cfg)

    rows = []
    for i, (_, r) in enumerate(result_df.iterrows()):
        rows.append({
            "dataset": dataset,
            "condition": condition_id,
            "epoch": epoch,
            "benchmark": "tvsd",
            "subject_idx": i,
            "layer": r["layer"],
            "score": round(r["score"], 4),
            "ci_low": round(r["ci_low"], 4) if r.get("ci_low") is not None else None,
            "ci_high": round(r["ci_high"], 4) if r.get("ci_high") is not None else None,
        })

    return rows


def print_summary(datasets, conditions, benchmarks):
    """Print results summary table."""
    if not os.path.exists(CSV_PATH):
        print("No results CSV found.")
        return

    df = pd.read_csv(CSV_PATH)

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
    parser.add_argument("--datasets", type=str, nargs="+", default=DATASETS,
                        choices=DATASETS)
    parser.add_argument("--conditions", type=int, nargs="+",
                        default=list(CONDITIONS.keys()),
                        choices=list(CONDITIONS.keys()))
    parser.add_argument("--epochs", type=int, nargs="+", default=EPOCHS)
    parser.add_argument("--benchmarks", type=str, nargs="+",
                        default=["things", "nsd", "tvsd"],
                        choices=["things", "nsd", "tvsd"])
    parser.add_argument("--force", action="store_true",
                        help="Re-evaluate even if result exists in CSV")
    parser.add_argument("--no_bootstrap", action="store_true")
    parser.add_argument("--print_only", action="store_true")
    args = parser.parse_args()

    if args.print_only:
        print_summary(args.datasets, args.conditions, args.benchmarks)
        return

    total = len(args.datasets) * len(args.conditions) * len(args.epochs) * len(args.benchmarks)
    completed, skipped = 0, 0
    existing = load_existing_results()

    for dataset in args.datasets:
        for condition_id in args.conditions:
            for epoch in args.epochs:
                for benchmark in args.benchmarks:
                    # Skip if checkpoint doesn't exist
                    if not checkpoint_exists(dataset, condition_id, epoch):
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

                    if benchmark == "things":
                        rows = eval_run(dataset, condition_id, epoch, benchmark)
                    elif benchmark == "nsd":
                        rows = eval_nsd(dataset, condition_id, epoch)
                    elif benchmark == "tvsd":
                        rows = eval_tvsd(dataset, condition_id, epoch)

                    save_results(rows)
                    completed += 1

                    # Release file handles between runs to avoid "Too many open files"
                    gc.collect()
                    torch.cuda.empty_cache()

    print(f"\nEvaluation complete. {completed}/{total} runs executed, {skipped} skipped.")
    print_summary(args.datasets, args.conditions, args.benchmarks)


if __name__ == "__main__":
    main()
