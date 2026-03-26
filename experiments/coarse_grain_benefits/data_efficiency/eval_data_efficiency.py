"""
Evaluate data-efficiency trained models on THINGS behavioral alignment
(and optionally NSD/TVSD). Uses the standard two-phase RSA pipeline
(layer selection on train, re-extract best layer on test).

Results are saved to per-architecture CSVs:
  - CustomCNN:     data_efficiency_results.csv
  - ResNet50:      data_efficiency_resnet50_results.csv
  - ViT:           data_efficiency_vit_base_results.csv
  - ConvNeXt:      data_efficiency_convnext_base_results.csv

Usage (from project root):
    # Run all architectures (resnet50, vit_base, convnext_base) on THINGS:
    python experiments/coarse_grain_benefits/data_efficiency/eval_data_efficiency.py --architecture all

    # Single architecture:
    python experiments/coarse_grain_benefits/data_efficiency/eval_data_efficiency.py --architecture resnet50

    # Include neural benchmarks:
    python experiments/coarse_grain_benefits/data_efficiency/eval_data_efficiency.py --architecture resnet50 --benchmarks things nsd tvsd

    # Print existing results:
    python experiments/coarse_grain_benefits/data_efficiency/eval_data_efficiency.py --architecture resnet50 --print_only
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
    SEED, SEED_LETTER, DEFAULT_PCA_LABELS, DATASETS, CHECKPOINT_BASE, ARCHITECTURES,
    get_conditions, get_checkpoint_dir, get_csv_path, save_results,
)


def _resolve_checkpoint_model(dataset, pca_labels, condition_id, arch, epoch):
    """Prefer BN-recalibrated checkpoint if available (needed for ResNet50)."""
    checkpoint_dir = get_checkpoint_dir(dataset, pca_labels, condition_id, arch=arch)
    base_path = os.path.join(CHECKPOINT_BASE, checkpoint_dir, f"cfg{condition_id}{SEED_LETTER}")
    recal = f"checkpoint_epoch_{epoch}_recal.pth"
    if os.path.exists(os.path.join(base_path, recal)):
        print(f"  [BN-recal] Using {recal}")
        return recal
    return f"checkpoint_epoch_{epoch}.pth"


def build_overrides(dataset, condition_id, epoch, benchmark, conditions, pca_labels,
                    arch="customcnn", region=None):
    """Build config overrides for a single eval run."""
    cond = conditions[condition_id]
    checkpoint_dir = os.path.join(
        CHECKPOINT_BASE, get_checkpoint_dir(dataset, pca_labels, condition_id, arch=arch))
    checkpoint_model = _resolve_checkpoint_model(dataset, pca_labels, condition_id, arch, epoch)

    overrides = [
        f"seed={SEED}",
        f"cfg_id={condition_id}",
        f"checkpoint_dir={checkpoint_dir}",
        f"checkpoint_model={checkpoint_model}",
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
        overrides.append(f"region={region or 'ventral visual stream'}")
        overrides.append("subject_idx=[0,1,2,3,4,5,6,7]")
    elif benchmark == "tvsd":
        overrides.append("neural_dataset=tvsd")
        overrides.append(f"region={region or 'IT'}")
        overrides.append("subject_idx=[0,1]")

    return overrides


def load_existing_results(csv_path):
    """Load existing CSV once and return set of completed (dataset, condition, epoch, benchmark, region) tuples."""
    if not os.path.exists(csv_path):
        return set()
    df = pd.read_csv(csv_path)
    # Backfill region for legacy rows that lack it
    if "region" not in df.columns:
        df["region"] = df["benchmark"].map(
            {"nsd": "ventral visual stream", "tvsd": "IT", "things": "N/A"})
    df["region"] = df["region"].fillna(
        df["benchmark"].map({"nsd": "ventral visual stream", "tvsd": "IT", "things": "N/A"}))
    return set(zip(df["dataset"], df["condition"], df["epoch"], df["benchmark"], df["region"]))


def result_exists(completed, dataset, condition_id, epoch, benchmark, region):
    """Check if result already exists using pre-loaded set."""
    return (dataset, condition_id, epoch, benchmark, region) in completed


def checkpoint_exists(dataset, condition_id, epoch, pca_labels, arch="customcnn"):
    """Check if the checkpoint file exists."""
    checkpoint_dir = get_checkpoint_dir(dataset, pca_labels, condition_id, arch=arch)
    path = os.path.join(CHECKPOINT_BASE, checkpoint_dir,
                        f"cfg{condition_id}{SEED_LETTER}",
                        f"checkpoint_epoch_{epoch}.pth")
    return os.path.exists(path)


def _make_result_rows(result_df, dataset, condition_id, epoch, benchmark, region):
    """Convert eval result DataFrame to list of row dicts for the CSV."""
    rows = []
    for i, (_, r) in enumerate(result_df.iterrows()):
        rows.append({
            "dataset": dataset,
            "condition": condition_id,
            "epoch": epoch,
            "benchmark": benchmark,
            "region": region,
            "subject_idx": "N/A" if benchmark == "things" else i,
            "layer": r["layer"],
            "score": round(r["score"], 4),
            "ci_low": round(r["ci_low"], 4) if r.get("ci_low") is not None else None,
            "ci_high": round(r["ci_high"], 4) if r.get("ci_high") is not None else None,
        })
    return rows


def eval_benchmark(dataset, condition_id, epoch, benchmark, conditions, pca_labels,
                   arch="customcnn", region=None):
    """Run a single evaluation and return result rows."""
    region_label = region or {"nsd": "ventral visual stream", "tvsd": "IT", "things": "N/A"}[benchmark]
    print(f"\n{'='*60}")
    print(f"Evaluating: {arch} | {condition_id}-class | {dataset} | epoch {epoch} | {benchmark} | {region_label}")
    print(f"{'='*60}")

    overrides = build_overrides(dataset, condition_id, epoch, benchmark, conditions, pca_labels,
                                arch=arch, region=region)
    cfg = load_config("configs/eval/base.json", overrides)
    cfg = validate_config(cfg)
    result_df = evals.eval(cfg)
    return _make_result_rows(result_df, dataset, condition_id, epoch, benchmark, region_label)


def print_summary(datasets, conditions, benchmarks, csv_path, arch="customcnn"):
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
        arch_label = arch.upper()
        if bench == "things":
            print(f"{arch_label} — THINGS Behavioral Alignment — Data Efficiency")
        elif bench == "nsd":
            print(f"{arch_label} — NSD Ventral Stream Alignment — Data Efficiency (mean across subjects)")
        elif bench == "tvsd":
            print(f"{arch_label} — TVSD IT Alignment — Data Efficiency (mean across subjects)")
        print(f"{'='*70}")
        print(f"{'Dataset':<22} {'Condition':>10} {'Epoch':>6} {'Score':>8}")
        print(f"{'-'*70}")

        for ds in datasets:
            ds_df = bdf[bdf["dataset"] == ds]
            if len(ds_df) == 0:
                continue
            for cond in conditions:
                cond_df = ds_df[ds_df["condition"] == cond]
                for epoch in sorted(cond_df["epoch"].unique()):
                    edf = cond_df[cond_df["epoch"] == epoch]
                    if len(edf) == 0:
                        continue
                    score = edf["score"].mean()
                    print(f"{ds:<22} {cond:>10} {epoch:>6} {score:>8.4f}")

        print(f"{'='*70}")


def _run_architecture(arch, args, conditions, bench_regions):
    """Run evaluation for a single architecture. Returns (completed, skipped) counts."""
    # When --epoch_per_dataset is used, each dataset gets exactly one epoch
    epoch_for_dataset = {
        "imagenet-mini-10": 100,
        "imagenet-mini-100": 50,
    }

    csv_path = get_csv_path(args.pca_labels, arch=arch)
    existing = load_existing_results(csv_path)
    completed, skipped = 0, 0

    for dataset in args.datasets:
        if args.epoch_per_dataset:
            epochs = [epoch_for_dataset[dataset]]
        else:
            epochs = args.epochs

        for condition_id in args.conditions:
            for epoch in epochs:
                for benchmark, region in bench_regions:
                    if not checkpoint_exists(dataset, condition_id, epoch,
                                             args.pca_labels, arch=arch):
                        print(f"[SKIP] {arch} {condition_id}-class {dataset} epoch {epoch} "
                              f"-- checkpoint not found")
                        skipped += 1
                        continue

                    if not args.force and result_exists(
                            existing, dataset, condition_id, epoch, benchmark, region):
                        print(f"[SKIP] {arch} {condition_id}-class {dataset} epoch {epoch} "
                              f"{benchmark}/{region} -- already evaluated")
                        skipped += 1
                        continue

                    rows = eval_benchmark(dataset, condition_id, epoch, benchmark,
                                          conditions, args.pca_labels,
                                          arch=arch,
                                          region=region if benchmark != "things" else None)
                    save_results(rows, csv_path)
                    completed += 1

                    gc.collect()
                    torch.cuda.empty_cache()

    total = completed + skipped
    print(f"\n{arch}: {completed}/{total} runs executed, {skipped} skipped.")
    print_summary(args.datasets, args.conditions, args.benchmarks, csv_path, arch=arch)
    return completed, skipped


# "all" runs the three non-CustomCNN architectures (the ones copied from Rockfish)
ALL_ARCHS = ["resnet50", "vit_base", "convnext_base"]


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate data efficiency models on THINGS (and optionally NSD/TVSD)")
    parser.add_argument("--architecture", type=str, default="all",
                        choices=list(ARCHITECTURES.keys()) + ["all"],
                        help="Model architecture, or 'all' for resnet50+vit_base+convnext_base (default: all)")
    parser.add_argument("--pca_labels", type=str, default=DEFAULT_PCA_LABELS,
                        help="PCA labels source, e.g. 'clip' or 'alexnet' (default: clip)")
    parser.add_argument("--datasets", type=str, nargs="+", default=["imagenet-mini-10"],
                        choices=DATASETS)
    parser.add_argument("--conditions", type=int, nargs="+", default=None,
                        help="Which conditions to evaluate (default: all)")
    parser.add_argument("--epochs", type=int, nargs="+", default=[100])
    parser.add_argument("--epoch_per_dataset", action="store_true",
                        help="Use dataset-specific epochs: mini-10->100, mini-100->50")
    parser.add_argument("--benchmarks", type=str, nargs="+",
                        default=["things"],
                        choices=["things", "nsd", "tvsd"])
    parser.add_argument("--nsd_regions", type=str, nargs="+",
                        default=["ventral visual stream"],
                        help="NSD regions to evaluate (default: ventral visual stream)")
    parser.add_argument("--tvsd_regions", type=str, nargs="+",
                        default=["IT"],
                        help="TVSD regions to evaluate (default: IT)")
    parser.add_argument("--force", action="store_true",
                        help="Re-evaluate even if result exists in CSV")
    parser.add_argument("--print_only", action="store_true")
    args = parser.parse_args()

    # Resolve architecture list
    archs = ALL_ARCHS if args.architecture == "all" else [args.architecture]

    conditions = get_conditions(args.pca_labels)
    if args.conditions is None:
        args.conditions = list(conditions.keys())

    # Build list of (benchmark, region) pairs
    bench_regions = []
    for bench in args.benchmarks:
        if bench == "nsd":
            for region in args.nsd_regions:
                bench_regions.append((bench, region))
        elif bench == "tvsd":
            for region in args.tvsd_regions:
                bench_regions.append((bench, region))
        else:
            bench_regions.append((bench, "N/A"))

    if args.print_only:
        for arch in archs:
            csv_path = get_csv_path(args.pca_labels, arch=arch)
            print_summary(args.datasets, args.conditions, args.benchmarks, csv_path, arch=arch)
        return

    total_completed, total_skipped = 0, 0
    for arch in archs:
        print(f"\n{'#'*60}")
        print(f"  Architecture: {arch}")
        print(f"{'#'*60}")
        c, s = _run_architecture(arch, args, conditions, bench_regions)
        total_completed += c
        total_skipped += s

    if len(archs) > 1:
        total = total_completed + total_skipped
        print(f"\nAll done. {total_completed}/{total} runs executed, {total_skipped} skipped.")


if __name__ == "__main__":
    main()
