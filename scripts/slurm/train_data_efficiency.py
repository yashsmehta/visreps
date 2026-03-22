"""
Submit Slurm jobs for the data efficiency experiment across multiple architectures.

Trains 4 architectures (CustomCNN, ResNet-50, ConvNeXt-Base, ViT-Base) on
imagenet-mini-{10,100} with 5 label granularities (8, 16, 32, 64 coarse + 1000),
using either CLIP or AlexNet PCA labels for coarse conditions.

Each architecture uses its own optimizer/LR/regularization from the existing
architecture config files, with epochs/warmup/checkpointing overridden uniformly.
Jobs are skipped if the final checkpoint already exists on disk.

Usage (run from project root on Rockfish):
    python scripts/slurm/train_data_efficiency.py --dry-run
    python scripts/slurm/train_data_efficiency.py
    python scripts/slurm/train_data_efficiency.py --pca_labels alexnet
    python scripts/slurm/train_data_efficiency.py --models ResNet50 ViTBase
    python scripts/slurm/train_data_efficiency.py --datasets imagenet-mini-10
    python scripts/slurm/train_data_efficiency.py --conditions 32 64 1000
"""

import argparse
import json
import os
import subprocess
from itertools import product
from pathlib import Path

# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

BASE_CONFIG = "configs/train/base.json"

MODELS = {
    "CustomCNN":     {"config": "configs/train/architectures/custom_cnn.json", "tag": "CNN",  "dir": "customcnn"},
    "ResNet50":      {"config": "configs/train/architectures/resnet50.json",   "tag": "RN50", "dir": "resnet50"},
    "ConvNeXt_Base": {"config": "configs/train/architectures/convnext_base.json", "tag": "CNX", "dir": "convnext_base"},
    "ViTBase":       {"config": "configs/train/architectures/vit_b_16.json",   "tag": "ViT",  "dir": "vitbase"},
}

DATASETS = ["imagenet-mini-10", "imagenet-mini-100"]
PCA_LABELS_CHOICES = ["clip", "alexnet"]

COARSE_CLASSES = [8, 16, 32, 64]
ALL_CONDITIONS = COARSE_CLASSES + [1000]

SEED = 1
SEED_LETTER = chr(ord("a") + SEED - 1)
NUM_EPOCHS = 100

SHARED_OVERRIDES = {
    "num_epochs": NUM_EPOCHS,
    "warmup_epochs": 10,
    "checkpoint_interval": 50,
    "log_checkpoints": True,
    "use_wandb": False,
}

SLURM_CONFIG = {
    "job-name": "visreps",
    "output": "scripts/slurm/slurm_logs/%j.out",
    "error": "scripts/slurm/slurm_logs/%j.err",
    "ntasks": "1",
    "cpus-per-task": "32",
    "gres": "gpu:1",
    "time": "10:00:00",
    "partition": "a100",
    "qos": "qos_gpu",
    "account": "mbonner5_gpu",
}

# =============================================================================
# INTERNAL
# =============================================================================


def get_checkpoint_dir(model_name, dataset, pca_labels, n_classes):
    """Checkpoint dir under data_efficiency/.

    Coarse: data_efficiency/{model}_{pca_labels}_{dataset}
    1000:   data_efficiency/{model}_{dataset}  (shared across PCA label runs)
    """
    base = MODELS[model_name]["dir"]
    if n_classes == 1000:
        return f"data_efficiency/{base}_{dataset}"
    return f"data_efficiency/{base}_{pca_labels}_{dataset}"


def final_checkpoint_exists(checkpoint_dir, n_classes):
    """Check if the final checkpoint for this condition exists on disk."""
    path = os.path.join(
        "model_checkpoints", checkpoint_dir,
        f"cfg{n_classes}{SEED_LETTER}",
        f"checkpoint_epoch_{NUM_EPOCHS}.pth",
    )
    return os.path.exists(path)


def make_job_name(model_name, dataset, n_classes, pca_labels):
    """Short Slurm job name, e.g. 'de_RN50_cl_m10_c32' or 'de_ViT_m100_std'."""
    tag = MODELS[model_name]["tag"]
    ds_tag = dataset.replace("imagenet-mini-", "m")
    if n_classes == 1000:
        return f"de_{tag}_{ds_tag}_std"
    pca_tag = pca_labels[:2]  # "cl" for clip, "al" for alexnet
    return f"de_{tag}_{pca_tag}_{ds_tag}_c{n_classes}"


def generate_slurm_script(base_config, arch_config, overrides, job_name):
    """Generate SLURM batch script content."""
    slurm_config = {**SLURM_CONFIG, "job-name": job_name}
    lines = ["#!/bin/bash"]
    lines += [f"#SBATCH --{k}={v}" for k, v in slurm_config.items()]
    lines += [
        "",
        "source .venv/bin/activate",
        'echo "Running on: $(hostname)"',
        "nvidia-smi",
        "",
        f"python -m visreps.run --mode train --config {base_config} {arch_config} "
        f"--override {' '.join(overrides)}",
        "deactivate",
    ]
    return "\n".join(lines)


def build_jobs(models, datasets, conditions, pca_labels, force=False):
    """Expand experiment grid into a flat list of job tuples."""
    pca_folder = f"pca_labels_{pca_labels}"
    jobs = []
    skipped = 0

    for model_name, dataset, n_classes in product(models, datasets, conditions):
        checkpoint_dir = get_checkpoint_dir(model_name, dataset, pca_labels, n_classes)

        if not force and final_checkpoint_exists(checkpoint_dir, n_classes):
            print(f"  [SKIP] {model_name} {dataset} {n_classes}-class — checkpoint exists")
            skipped += 1
            continue

        arch_config = MODELS[model_name]["config"]
        is_coarse = n_classes != 1000

        overrides = [
            f"seed={SEED}",
            f"dataset={dataset}",
            f"checkpoint_dir={checkpoint_dir}",
            f"pca_labels={json.dumps(is_coarse)}",
        ]
        if is_coarse:
            overrides.append(f"pca_n_classes={n_classes}")
            overrides.append(f"pca_labels_folder={pca_folder}")

        for k, v in SHARED_OVERRIDES.items():
            overrides.append(f"{k}={json.dumps(v) if isinstance(v, bool) else v}")

        job_name = make_job_name(model_name, dataset, n_classes, pca_labels)
        jobs.append((model_name, dataset, n_classes, arch_config, overrides, job_name))

    return jobs, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Submit data efficiency training jobs to Slurm")
    parser.add_argument("--dry-run", action="store_true",
                        help="List jobs without submitting")
    parser.add_argument("--pca_labels", default="clip", choices=PCA_LABELS_CHOICES,
                        help="PCA label source (default: clip)")
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()),
                        choices=list(MODELS.keys()),
                        help="Which models to train (default: all)")
    parser.add_argument("--datasets", nargs="+", default=DATASETS,
                        choices=DATASETS,
                        help="Which datasets to train on (default: all)")
    parser.add_argument("--conditions", type=int, nargs="+",
                        default=ALL_CONDITIONS, choices=ALL_CONDITIONS,
                        help="Which conditions to train (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="Submit even if checkpoint exists")
    args = parser.parse_args()

    jobs, skipped = build_jobs(args.models, args.datasets, args.conditions,
                               args.pca_labels, args.force)

    if args.dry_run:
        print(f"\n{'#':>3}  {'Job Name':<26}  {'Model':<14}  {'Dataset':<18}  {'Classes':>7}")
        print(f"{'─'*3}  {'─'*26}  {'─'*14}  {'─'*18}  {'─'*7}")
        for i, (model, dataset, n_classes, _, _, job_name) in enumerate(jobs, 1):
            print(f"{i:>3}  {job_name:<26}  {model:<14}  {dataset:<18}  {n_classes:>7}")
        print(f"\nTOTAL={len(jobs)} to submit, {skipped} skipped (checkpoint exists)")
        return

    Path("scripts/slurm/slurm_logs").mkdir(parents=True, exist_ok=True)
    Path("scripts/slurm/tmp").mkdir(parents=True, exist_ok=True)

    print(f"\nSubmitting {len(jobs)} Slurm jobs ({skipped} skipped)\n")

    submitted = 0
    for i, (model, dataset, n_classes, arch_config, overrides, job_name) in enumerate(jobs, 1):
        script_path = f"scripts/slurm/tmp/data_eff_{i}.sh"

        with open(script_path, "w") as f:
            f.write(generate_slurm_script(BASE_CONFIG, arch_config, overrides, job_name))

        print(f"  [{i}/{len(jobs)}] {job_name}  ({model}, {dataset}, {n_classes}-class)")

        result = subprocess.run(["sbatch", script_path])
        if result.returncode == 0:
            os.remove(script_path)
            submitted += 1
        else:
            print(f"    sbatch failed — script kept at {script_path}")

    print(f"\nDone. {submitted}/{len(jobs)} jobs submitted, {skipped} skipped.")


if __name__ == "__main__":
    main()
