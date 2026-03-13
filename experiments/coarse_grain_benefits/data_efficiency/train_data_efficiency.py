"""
Train CustomCNN on imagenet-mini subsets to test data efficiency hypothesis:
Does coarse-grained supervision (8 CLIP classes) produce more behaviorally
aligned representations than 1000-class supervision, even with limited data?

Usage (from project root):
    python experiments/coarse_grain_benefits/data_efficiency/train_data_efficiency.py --dataset imagenet-mini-50
    python experiments/coarse_grain_benefits/data_efficiency/train_data_efficiency.py --dataset imagenet-mini-10
    python experiments/coarse_grain_benefits/data_efficiency/train_data_efficiency.py --dataset imagenet-mini-10 --conditions 8
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

TRAINING_PARAMS = {
    "batchsize": 256,
    "learning_rate": 0.008,
    "num_epochs": 200,
    "warmup_epochs": 20,
    "checkpoint_interval": 50,
    "log_interval": 200,
    "log_checkpoints": True,
    "use_wandb": False,
    "model_class": "custom_model",
}

CONDITIONS = {
    8: {"pca_labels": True, "pca_n_classes": 8, "pca_labels_folder": "pca_labels_clip"},
    1000: {"pca_labels": False, "pca_n_classes": 1000},
}


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
    print(f"\n{n_classes}-class training complete.")


def main():
    parser = argparse.ArgumentParser(description="Train models for data efficiency experiment")
    parser.add_argument("--dataset", type=str, default="imagenet-mini-50",
                        choices=["imagenet-mini-10", "imagenet-mini-50", "imagenet-mini-200"],
                        help="Dataset to train on")
    parser.add_argument("--conditions", type=int, nargs="+", default=[8, 1000],
                        choices=[8, 1000], help="Which conditions to train (default: both)")
    args = parser.parse_args()

    for n_classes in args.conditions:
        train_condition(n_classes, args.dataset)

    print("\nAll training complete. Run eval_data_efficiency.py to evaluate on THINGS.")


if __name__ == "__main__":
    main()
