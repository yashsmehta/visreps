"""
Recalibrate BatchNorm running statistics for ResNet50 data-efficiency checkpoints.

ResNet50 models trained on few-class tasks can develop miscalibrated BN running
statistics, causing representational collapse at eval time (see experiments/bn_recalibration/).
This script recalibrates by doing one full forward pass through imagenet-mini-10
with momentum=None (cumulative average), then saves as checkpoint_epoch_{N}_recal.pth.

Usage (from project root):
    python experiments/coarse_grain_benefits/data_efficiency/recalibrate_resnet50.py
    python experiments/coarse_grain_benefits/data_efficiency/recalibrate_resnet50.py --conditions 16 32
    python experiments/coarse_grain_benefits/data_efficiency/recalibrate_resnet50.py --force
"""

import os
import sys
import time
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

import torch
from omegaconf import OmegaConf

from visreps.config import ConfigDict
from visreps.dataloaders.obj_cls import get_obj_cls_loader
from visreps.models.utils import load_model, TORCHVISION_RETURN_NODES
from visreps.utils import get_seed_letter

from experiments.coarse_grain_benefits.data_efficiency.shared import (
    SEED, SEED_LETTER, CHECKPOINT_BASE,
    get_conditions, get_checkpoint_dir,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET = "imagenet-mini-10"
EPOCH = 100


def recalibrate_bn(model, n_batches=None):
    """Recalibrate BN running stats on imagenet-mini-10 using cumulative average.

    One full pass through the training set (~39 batches of 256). With momentum=None,
    this computes the exact sample mean/variance over all 10K images.
    """
    raw = model.model if hasattr(model, "model") else model

    # Reset and switch to cumulative average
    bn_count = 0
    raw.train()
    for m in raw.modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            m.reset_running_stats()
            m.momentum = None
            bn_count += 1

    print(f"    Reset {bn_count} BN layers, running forward pass...")

    cal_cfg = ConfigDict({
        "dataset": DATASET,
        "batchsize": 256,
        "num_workers": 8,
        "data_augment": False,
        "pca_labels": False,
        "pca_n_classes": 1000,
    })
    _, loaders = get_obj_cls_loader(cal_cfg)

    with torch.no_grad():
        for i, (imgs, _) in enumerate(loaders["train"]):
            raw(imgs.to(DEVICE))
            if n_batches and i + 1 >= n_batches:
                break

    raw.eval()
    n_done = i + 1
    print(f"    Recalibrated on {n_done} batches ({n_done * 256 // 1000}K images)")


def build_cfg(cfg_id, checkpoint_dir, pca_labels, pca_n_classes):
    """Build a minimal config for loading the model."""
    seed_letter = get_seed_letter(SEED)
    train_cfg_path = os.path.join(
        CHECKPOINT_BASE, checkpoint_dir,
        f"cfg{cfg_id}{seed_letter}", "config.json"
    )
    base = OmegaConf.load(train_cfg_path)

    overrides = OmegaConf.create({
        "mode": "eval",
        "seed": SEED,
        "load_model_from": "checkpoint",
        "cfg_id": cfg_id,
        "checkpoint_dir": os.path.join(CHECKPOINT_BASE, checkpoint_dir),
        "checkpoint_model": f"checkpoint_epoch_{EPOCH}.pth",
        "model_class": "standard_model",
        "model_name": "ResNet50",
        "return_nodes": list(TORCHVISION_RETURN_NODES["ResNet50"]),
    })

    cfg = OmegaConf.merge(base, overrides)
    return cfg


def main():
    parser = argparse.ArgumentParser(
        description="Recalibrate BN stats for ResNet50 data-efficiency checkpoints")
    parser.add_argument("--conditions", type=int, nargs="+", default=None,
                        help="Which conditions to recalibrate (default: all)")
    parser.add_argument("--pca_labels", type=str, default="clip")
    parser.add_argument("--force", action="store_true",
                        help="Re-recalibrate even if _recal.pth exists")
    args = parser.parse_args()

    conditions = get_conditions(args.pca_labels)
    if args.conditions is None:
        args.conditions = list(conditions.keys())

    completed, skipped = 0, 0

    for cfg_id in args.conditions:
        cond = conditions[cfg_id]
        checkpoint_dir = get_checkpoint_dir(DATASET, args.pca_labels, cfg_id, arch="resnet50")
        ckpt_path = os.path.join(
            CHECKPOINT_BASE, checkpoint_dir,
            f"cfg{cfg_id}{SEED_LETTER}",
            f"checkpoint_epoch_{EPOCH}.pth"
        )
        recal_path = os.path.join(
            CHECKPOINT_BASE, checkpoint_dir,
            f"cfg{cfg_id}{SEED_LETTER}",
            f"checkpoint_epoch_{EPOCH}_recal.pth"
        )

        if not os.path.exists(ckpt_path):
            print(f"[SKIP] cfg{cfg_id} — checkpoint not found: {ckpt_path}")
            skipped += 1
            continue

        if not args.force and os.path.exists(recal_path):
            print(f"[SKIP] cfg{cfg_id} — _recal.pth already exists")
            skipped += 1
            continue

        print(f"\n{'='*60}")
        print(f"  Recalibrating cfg{cfg_id} ({cfg_id}-class ResNet50)")
        print(f"{'='*60}")

        # Load model
        cfg = build_cfg(cfg_id, checkpoint_dir,
                        cond["pca_labels"], cond["pca_n_classes"])
        model = load_model(cfg, DEVICE, verbose=False)

        # Recalibrate
        t0 = time.time()
        recalibrate_bn(model)
        elapsed = time.time() - t0
        print(f"    Took {elapsed:.1f}s")

        # Save: load original checkpoint, replace model object with recalibrated one
        raw = model.model if hasattr(model, "model") else model
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        checkpoint["model"] = raw.cpu()
        torch.save(checkpoint, recal_path)
        size_mb = os.path.getsize(recal_path) / 1e6
        print(f"    Saved: {recal_path} ({size_mb:.0f} MB)")
        completed += 1

        del model
        torch.cuda.empty_cache()

    total = completed + skipped
    print(f"\nDone. {completed}/{total} recalibrated, {skipped} skipped.")


if __name__ == "__main__":
    main()
