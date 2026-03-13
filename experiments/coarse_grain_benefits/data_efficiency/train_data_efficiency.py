"""
Train CustomCNN on imagenet-mini subsets for the data efficiency experiment.
Tests whether coarse-grained supervision produces more aligned representations
than fine-grained supervision, even with limited data.

Trains 5 conditions (8, 16, 32, 64 CLIP classes + 1000-class) on 3 datasets.
Skips models whose final checkpoint already exists.

Usage (from project root):
    python experiments/coarse_grain_benefits/data_efficiency/train_data_efficiency.py
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

SEED = 1
SEED_LETTER = "a"
NUM_EPOCHS = 200

DATASETS = ["imagenet-mini-5", "imagenet-mini-10", "imagenet-mini-50"]

CONDITIONS = {
    8:    {"pca_labels": True, "pca_n_classes": 8,  "pca_labels_folder": "pca_labels_clip"},
    16:   {"pca_labels": True, "pca_n_classes": 16, "pca_labels_folder": "pca_labels_clip"},
    32:   {"pca_labels": True, "pca_n_classes": 32, "pca_labels_folder": "pca_labels_clip"},
    64:   {"pca_labels": True, "pca_n_classes": 64, "pca_labels_folder": "pca_labels_clip"},
    1000: {"pca_labels": False, "pca_n_classes": 1000},
}

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


def checkpoint_exists(dataset, n_classes):
    """Check if the final checkpoint for this condition already exists."""
    checkpoint_dir = f"model_checkpoints/data_efficiency_{dataset}"
    path = os.path.join(checkpoint_dir, f"cfg{n_classes}{SEED_LETTER}",
                        f"checkpoint_epoch_{NUM_EPOCHS}.pth")
    return os.path.exists(path)


def train_condition(n_classes, dataset):
    """Train a single condition."""
    condition = CONDITIONS[n_classes]
    checkpoint_dir = f"data_efficiency_{dataset}"
    print(f"\n{'='*60}")
    print(f"Training {n_classes}-class model on {dataset}")
    print(f"Checkpoints: model_checkpoints/{checkpoint_dir}/")
    print(f"{'='*60}\n")

    overrides = []
    params = {**TRAINING_PARAMS, **condition, "seed": SEED,
              "dataset": dataset, "checkpoint_dir": checkpoint_dir}
    for k, v in params.items():
        overrides.append(f"{k}={v}")

    cfg = load_config("configs/train/base.json", overrides)
    cfg = validate_config(cfg)
    Trainer(cfg).train()
    print(f"\n{n_classes}-class training on {dataset} complete.")


def main():
    parser = argparse.ArgumentParser(description="Train models for data efficiency experiment")
    parser.add_argument("--datasets", type=str, nargs="+", default=DATASETS,
                        choices=DATASETS, help="Datasets to train on")
    parser.add_argument("--conditions", type=int, nargs="+",
                        default=list(CONDITIONS.keys()),
                        choices=list(CONDITIONS.keys()),
                        help="Which conditions to train")
    parser.add_argument("--force", action="store_true",
                        help="Train even if checkpoint exists")
    args = parser.parse_args()

    total = len(args.datasets) * len(args.conditions)
    skipped = 0

    for dataset in args.datasets:
        for n_classes in args.conditions:
            if not args.force and checkpoint_exists(dataset, n_classes):
                print(f"[SKIP] {n_classes}-class on {dataset} — checkpoint exists")
                skipped += 1
                continue
            train_condition(n_classes, dataset)

    print(f"\nTraining complete. {total - skipped}/{total} runs executed, {skipped} skipped.")
    print("Run eval_data_efficiency.py to evaluate.")


if __name__ == "__main__":
    main()
