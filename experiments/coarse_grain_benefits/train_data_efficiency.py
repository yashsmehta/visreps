"""
Train CustomCNN on imagenet-mini-50 to test data efficiency hypothesis:
Does coarse-grained supervision (8 CLIP classes) produce more behaviorally
aligned representations than 1000-class supervision, even with limited data?

Usage (from project root):
    python experiments/coarse_grain_benefits/train_data_efficiency.py
    python experiments/coarse_grain_benefits/train_data_efficiency.py --conditions 8  # only 8-class
    python experiments/coarse_grain_benefits/train_data_efficiency.py --conditions 1000  # only 1000-class
"""

import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

from visreps.trainer import Trainer
from visreps.utils import load_config, validate_config

CHECKPOINT_DIR = "data_efficiency"
SEED = 1

# Training hyperparams for imagenet-mini-50
OVERRIDES = {
    "dataset": "imagenet-mini-50",
    "batchsize": 256,
    "learning_rate": 0.008,
    "num_epochs": 200,
    "warmup_epochs": 20,
    "checkpoint_interval": 50,
    "log_interval": 200,
    "log_checkpoints": True,
    "use_wandb": False,
    "checkpoint_dir": CHECKPOINT_DIR,
    "model_class": "custom_model",
}

CONDITIONS = {
    8: {"pca_labels": True, "pca_n_classes": 8, "pca_labels_folder": "pca_labels_clip"},
    1000: {"pca_labels": False, "pca_n_classes": 1000},
}


def train_condition(n_classes):
    """Train a single condition."""
    condition = CONDITIONS[n_classes]
    print(f"\n{'='*60}")
    print(f"Training {n_classes}-class model on imagenet-mini-50")
    print(f"{'='*60}\n")

    overrides = []
    for k, v in {**OVERRIDES, **condition, "seed": SEED}.items():
        overrides.append(f"{k}={v}")

    cfg = load_config("configs/train/base.json", overrides)
    cfg = validate_config(cfg)
    Trainer(cfg).train()
    print(f"\n{n_classes}-class training complete.")


def main():
    parser = argparse.ArgumentParser(description="Train models for data efficiency experiment")
    parser.add_argument("--conditions", type=int, nargs="+", default=[8, 1000],
                        choices=[8, 1000], help="Which conditions to train (default: both)")
    args = parser.parse_args()

    for n_classes in args.conditions:
        train_condition(n_classes)

    print("\nAll training complete. Run eval_data_efficiency.py to evaluate on THINGS.")


if __name__ == "__main__":
    main()
