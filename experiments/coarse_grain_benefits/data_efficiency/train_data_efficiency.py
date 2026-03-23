"""
Train CustomCNN on imagenet-mini subsets for the data efficiency experiment.
Tests whether coarse-grained supervision produces more aligned representations
than fine-grained supervision, even with limited data.

Trains 5 conditions (8, 16, 32, 64 coarse classes + 1000-class) on 3 datasets.
Skips models whose final checkpoint already exists.

Usage (from project root):
    python experiments/coarse_grain_benefits/data_efficiency/train_data_efficiency.py
    python experiments/coarse_grain_benefits/data_efficiency/train_data_efficiency.py --pca_labels alexnet
    python experiments/coarse_grain_benefits/data_efficiency/train_data_efficiency.py --datasets imagenet-mini-5
    python experiments/coarse_grain_benefits/data_efficiency/train_data_efficiency.py --conditions 16 32 64
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

from visreps.trainer import Trainer
from visreps.utils import load_config, validate_config
from experiments.coarse_grain_benefits.data_efficiency.shared import (
    SEED, SEED_LETTER, NUM_EPOCHS, DEFAULT_PCA_LABELS, DATASETS, CHECKPOINT_BASE,
    get_conditions, get_checkpoint_dir,
)

TRAINING_PARAMS = {
    "batchsize": 256,
    "learning_rate": 0.008,
    "num_epochs": NUM_EPOCHS,
    "warmup_epochs": 20,
    "checkpoint_interval": 50,
    "log_interval": 200,
    "log_checkpoints": True,
    "use_wandb": False,
    "model_class": "custom_model",
}


def checkpoint_exists(dataset, n_classes, pca_labels):
    """Check if the final checkpoint for this condition already exists."""
    checkpoint_dir = get_checkpoint_dir(dataset, pca_labels, n_classes)
    path = os.path.join(CHECKPOINT_BASE, checkpoint_dir,
                        f"cfg{n_classes}{SEED_LETTER}",
                        f"checkpoint_epoch_{NUM_EPOCHS}.pth")
    return os.path.exists(path)


def train_condition(n_classes, dataset, conditions, pca_labels):
    """Train a single condition."""
    condition = conditions[n_classes]
    checkpoint_dir = get_checkpoint_dir(dataset, pca_labels, n_classes)
    print(f"\n{'='*60}")
    print(f"Training {n_classes}-class model on {dataset}")
    print(f"Checkpoints: {CHECKPOINT_BASE}/{checkpoint_dir}/")
    print(f"{'='*60}\n")

    overrides = []
    params = {**TRAINING_PARAMS, **condition, "seed": SEED,
              "dataset": dataset, "checkpoint_dir": checkpoint_dir}
    for k, v in params.items():
        overrides.append(f"{k}={v}")

    cfg = load_config(["configs/train/base.json", "configs/train/architectures/custom_cnn.json"], overrides)
    cfg = validate_config(cfg)
    Trainer(cfg).train()
    print(f"\n{n_classes}-class training on {dataset} complete.")


def main():
    parser = argparse.ArgumentParser(description="Train models for data efficiency experiment")
    parser.add_argument("--pca_labels", type=str, default=DEFAULT_PCA_LABELS,
                        help="PCA labels source, e.g. 'clip' or 'alexnet' (default: clip)")
    parser.add_argument("--datasets", type=str, nargs="+", default=DATASETS,
                        choices=DATASETS, help="Datasets to train on")
    parser.add_argument("--conditions", type=int, nargs="+", default=None,
                        help="Which conditions to train (default: all)")
    parser.add_argument("--force", action="store_true",
                        help="Train even if checkpoint exists")
    args = parser.parse_args()

    conditions = get_conditions(args.pca_labels)
    if args.conditions is None:
        args.conditions = list(conditions.keys())

    total = len(args.datasets) * len(args.conditions)
    skipped = 0

    for dataset in args.datasets:
        for n_classes in args.conditions:
            if not args.force and checkpoint_exists(dataset, n_classes, args.pca_labels):
                print(f"[SKIP] {n_classes}-class on {dataset} — checkpoint exists")
                skipped += 1
                continue
            train_condition(n_classes, dataset, conditions, args.pca_labels)

    print(f"\nTraining complete. {total - skipped}/{total} runs executed, {skipped} skipped.")
    print("Run eval_data_efficiency.py to evaluate.")


if __name__ == "__main__":
    main()
