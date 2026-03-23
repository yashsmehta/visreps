"""
BatchNorm Recalibration for ResNet50 Models
============================================

Recomputes BN running statistics on ImageNet after training, then runs
the exact THINGS-behavior evaluation pipeline (SRP extraction, 80/20
concept split, layer selection, re-extraction without SRP, Spearman RDM).

Usage:
    # Single model
    python experiments/bn_recalibration/recalibrate_bn.py --cfg_id 16

    # All coarseness levels
    python experiments/bn_recalibration/recalibrate_bn.py --all

    # Custom checkpoint
    python experiments/bn_recalibration/recalibrate_bn.py \
        --cfg_id 16 \
        --checkpoint_dir /data/ymehta3/resnet50_clip_pca \
        --epoch 20 \
        --n_batches 2000

Must be run from the project root with .env sourced.
"""

import argparse
import time

import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision import transforms

from visreps.analysis.alignment import AlignmentData, prepare_concept_alignment
from visreps.analysis.rsa import compute_rsa
from visreps.config import ConfigDict
from visreps.dataloaders.neural import _make_loader, load_things_data
from visreps.dataloaders.obj_cls import get_obj_cls_loader
from visreps.models.utils import configure_feature_extractor, get_activations, load_model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

THINGS_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Default model configs: cfg_id -> (checkpoint_dir, pca_labels, pca_n_classes)
MODEL_CONFIGS = {
    2:    ("/data/ymehta3/resnet50_clip_pca", True,  2),
    4:    ("/data/ymehta3/resnet50_clip_pca", True,  4),
    8:    ("/data/ymehta3/resnet50_clip_pca", True,  8),
    16:   ("/data/ymehta3/resnet50_clip_pca", True,  16),
    32:   ("/data/ymehta3/resnet50_clip_pca", True,  32),
    64:   ("/data/ymehta3/resnet50_clip_pca", True,  64),
    1000: ("/data/ymehta3/resnet50_default",  False, 1000),
}


def recalibrate_bn(model, pca_labels, pca_n_classes, n_batches=2000):
    """
    Recalibrate BN running statistics on ImageNet.

    Resets running_mean / running_var, then does a single forward pass
    through `n_batches` of ImageNet (batch_size=256) using a cumulative
    moving average (momentum=None) so every image contributes equally.

    Args:
        model: FeatureExtractor wrapping a ResNet50 (or the raw nn.Module).
        pca_labels: Whether to apply PCA label mapping for the dataloader.
        pca_n_classes: Number of PCA classes (only used if pca_labels=True).
        n_batches: Number of calibration batches (~256 images each).
    """
    raw = model.model if hasattr(model, "model") else model

    # Reset and switch to cumulative average
    raw.train()
    for m in raw.modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            m.reset_running_stats()
            m.momentum = None  # cumulative average instead of EMA

    cal_cfg = ConfigDict({
        "dataset": "imagenet",
        "batchsize": 256,
        "num_workers": 16,
        "data_augment": False,
        "pca_labels": pca_labels,
        "pca_n_classes": pca_n_classes,
        "pca_labels_folder": "pca_labels_clip",
    })
    _, loaders = get_obj_cls_loader(cal_cfg)

    with torch.no_grad():
        for i, (imgs, _) in enumerate(loaders["train"]):
            raw(imgs.to(DEVICE))
            if (i + 1) % 500 == 0:
                print(f"    Calibration batch {i + 1}/{n_batches}")
            if i + 1 >= n_batches:
                break

    raw.eval()
    print(f"    BN recalibrated on {min(n_batches, i + 1)} batches "
          f"({min(n_batches, i + 1) * 256 // 1000}K images)")


def build_eval_cfg(cfg_id, checkpoint_dir, pca_labels, pca_n_classes, epoch):
    return OmegaConf.create({
        "mode": "eval",
        "seed": 1,
        "load_model_from": "checkpoint",
        "cfg_id": cfg_id,
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_model": f"checkpoint_epoch_{epoch}.pth",
        "pca_labels": pca_labels,
        "pca_n_classes": pca_n_classes,
        "pca_labels_folder": "pca_labels_clip",
        "epoch": epoch,
        "model_class": "standard_model",
        "model_name": "ResNet50",
        "pretrained_dataset": "none",
        "neural_dataset": "things-behavior",
        "analysis": "rsa",
        "compare_method": "spearman",
        "bootstrap": False,
        "return_nodes": ["conv1"]
                        + [f"block{i}" for i in range(1, 17, 2)]
                        + ["block16"],
        "reconstruct_from_pcs": False,
        "dataset": "imagenet",
        "batchsize": 64,
        "num_workers": 16,
        "data_augment": False,
        "region": "N/A",
        "subject_idx": "N/A",
    })


def run_things_eval(cfg, model, stimuli, neural_data_raw):
    """Run exact THINGS eval: SRP → concept avg → 80/20 split → RSA."""
    dl = _make_loader(stimuli, THINGS_TRANSFORM, batch=64, workers=16)
    acts, ids = get_activations(model, dl, DEVICE)
    all_concepts = prepare_concept_alignment(cfg, acts, neural_data_raw, ids)

    # 80/20 concept split (seed=42, matching production pipeline)
    rng = np.random.RandomState(42)
    n = all_concepts.neural.size(0)
    perm = rng.permutation(n)
    n_sel = int(n * 0.2)
    sel_idx, eval_idx = perm[:n_sel], perm[n_sel:]

    selection = AlignmentData(
        activations={l: a[sel_idx] for l, a in all_concepts.activations.items()},
        neural=all_concepts.neural[sel_idx],
        stimulus_ids=[all_concepts.stimulus_ids[i] for i in sel_idx],
    )
    evaluation = AlignmentData(
        activations={l: a[eval_idx] for l, a in all_concepts.activations.items()},
        neural=all_concepts.neural[eval_idx],
        stimulus_ids=[all_concepts.stimulus_ids[i] for i in eval_idx],
        concept_image_ids={
            all_concepts.stimulus_ids[i]:
                all_concepts.concept_image_ids[all_concepts.stimulus_ids[i]]
            for i in eval_idx
        },
    )

    # Build image → concept map for re-extraction
    img_to_concepts = {}
    for concept, img_ids in evaluation.concept_image_ids.items():
        for sid in img_ids:
            img_to_concepts.setdefault(str(sid), []).append(concept)

    def re_extract_fn(layer, sids=None):
        raw = model.model if hasattr(model, "model") else model
        raw.eval()
        dl_re = _make_loader(stimuli, THINGS_TRANSFORM, batch=64, workers=16)
        sums, counts = {}, {}
        with torch.no_grad():
            for imgs, keys in dl_re:
                feats = model(imgs.to(DEVICE))
                out = feats[layer].reshape(feats[layer].size(0), -1).cpu().float()
                for i, key in enumerate(keys):
                    for c in img_to_concepts.get(str(key), []):
                        if c not in sums:
                            sums[c] = torch.zeros(out.size(1))
                            counts[c] = 0
                        sums[c] += out[i]
                        counts[c] += 1
        avgs = [
            sums[c] / counts[c] if c in sums and counts[c] > 0
            else torch.zeros(next(iter(sums.values())).size(0))
            for c in evaluation.stimulus_ids
        ]
        return torch.stack(avgs), evaluation.stimulus_ids

    results = compute_rsa(
        cfg, selection, evaluation, verbose=True, re_extract_fn=re_extract_fn
    )
    r = results[0]
    return r["score"], r["layer"]


def main():
    parser = argparse.ArgumentParser(description="BN recalibration + THINGS eval")
    parser.add_argument("--cfg_id", type=int, help="Single cfg_id to evaluate")
    parser.add_argument("--all", action="store_true", help="Run all coarseness levels")
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--n_batches", type=int, default=2000,
                        help="Calibration batches (256 imgs each, default 2000 = ~512K)")
    args = parser.parse_args()

    if not args.all and args.cfg_id is None:
        parser.error("Specify --cfg_id or --all")

    cfg_ids = sorted(MODEL_CONFIGS.keys()) if args.all else [args.cfg_id]

    # Load THINGS once
    print("Loading THINGS data...")
    neural_data_raw, stimuli = load_things_data()

    results = []
    for cfg_id in cfg_ids:
        if cfg_id not in MODEL_CONFIGS and args.checkpoint_dir is None:
            print(f"  Unknown cfg_id={cfg_id}, provide --checkpoint_dir")
            continue

        ckpt_dir, pca, n_cls = MODEL_CONFIGS.get(
            cfg_id, (args.checkpoint_dir, cfg_id != 1000, cfg_id)
        )
        if args.checkpoint_dir:
            ckpt_dir = args.checkpoint_dir

        print(f"\n{'=' * 60}")
        print(f"  cfg{cfg_id} — BN recalibration")
        print(f"{'=' * 60}")

        cfg = build_eval_cfg(cfg_id, ckpt_dir, pca, n_cls, args.epoch)
        model = load_model(cfg, DEVICE, verbose=False)
        model = configure_feature_extractor(cfg, model, verbose=False)

        print("  Recalibrating BN...")
        recalibrate_bn(model, pca, n_cls, n_batches=args.n_batches)

        t0 = time.time()
        score, layer = run_things_eval(cfg, model, stimuli, neural_data_raw)
        elapsed = time.time() - t0

        print(f"\n  Result: score={score:.4f}  layer={layer}  ({elapsed:.0f}s)")
        results.append({"cfg_id": cfg_id, "score": score, "layer": layer})

        del model
        torch.cuda.empty_cache()

    # Summary
    if len(results) > 1:
        print(f"\n\n{'=' * 50}")
        print("SUMMARY — Recalibrated BN THINGS scores")
        print(f"{'=' * 50}")
        print(f"{'cfg_id':>6} | {'Score':>8} | {'Layer':>8}")
        print("-" * 30)
        for r in results:
            print(f"{r['cfg_id']:>6} | {r['score']:>8.4f} | {r['layer']:>8}")


if __name__ == "__main__":
    main()
