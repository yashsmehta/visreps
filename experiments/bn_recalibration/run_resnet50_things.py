"""
BN Recalibration + THINGS Evaluation for All ResNet50 Models
============================================================

Recalibrates BatchNorm running statistics on ImageNet, then runs the
exact same THINGS-behavior RSA evaluation pipeline as evals.py
(with bootstrap=True). Results are saved to results.db, overriding
any existing entries for the same (cfg_id, epoch, seed, ...) combination.

Usage:
    # All models (default)
    python experiments/bn_recalibration/run_all_things.py

    # Specific models only
    python experiments/bn_recalibration/run_all_things.py --cfg_ids 16 1000

    # Fewer calibration batches (faster, less accurate)
    python experiments/bn_recalibration/run_all_things.py --n_batches 500

Must be run from the project root with .env sourced.
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

from visreps.analysis.alignment import (
    AlignmentData,
    compute_traintest_alignment,
    prepare_concept_alignment,
)
from visreps.analysis.reconstruct_from_pcs import reconstruct_from_pcs
from visreps.config import ConfigDict
from visreps.dataloaders.neural import load_things_data, _make_loader
from visreps.dataloaders.obj_cls import get_obj_cls_loader, get_transform
from visreps.models.utils import (
    TORCHVISION_RETURN_NODES,
    configure_feature_extractor,
    get_activations,
    load_model,
)
from visreps.utils import get_seed_letter, rprint, save_results


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# cfg_id -> (checkpoint_dir, pca_labels, pca_n_classes, eval_epoch)
MODEL_CONFIGS = {
    2:    ("/data/ymehta3/resnet50_clip_pca", True,  2,    20),
    4:    ("/data/ymehta3/resnet50_clip_pca", True,  4,    20),
    8:    ("/data/ymehta3/resnet50_clip_pca", True,  8,    20),
    16:   ("/data/ymehta3/resnet50_clip_pca", True,  16,   20),
    32:   ("/data/ymehta3/resnet50_clip_pca", True,  32,   20),
    64:   ("/data/ymehta3/resnet50_clip_pca", True,  64,   20),
    1000: ("/data/ymehta3/resnet50_default",  False, 1000, 20),
}


def recalibrate_bn(model, pca_labels, pca_n_classes, n_batches=2000):
    """Recalibrate BN running stats on ImageNet using cumulative average.

    Resets running_mean / running_var, sets momentum=None (cumulative
    average so every image contributes equally), then runs a single
    forward pass through n_batches of ImageNet (batch_size=256).
    """
    raw = model.model if hasattr(model, "model") else model

    raw.train()
    for m in raw.modules():
        if isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm1d)):
            m.reset_running_stats()
            m.momentum = None  # cumulative average, not EMA

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
    n_done = min(n_batches, i + 1)
    print(f"    BN recalibrated on {n_done} batches ({n_done * 256 // 1000}K images)")


def build_eval_cfg(cfg_id, checkpoint_dir, pca_labels, pca_n_classes, epoch):
    """Build an eval config that matches what evals.eval() would produce.

    Loads the training config from the checkpoint directory and merges
    eval-specific overrides on top — same as evals._load_cfg() does.
    This ensures the run_id hash matches the standard pipeline.
    """
    seed_letter = get_seed_letter(1)
    train_cfg_path = f"{checkpoint_dir}/cfg{cfg_id}{seed_letter}/config.json"
    base = OmegaConf.load(train_cfg_path)
    base.epoch = epoch
    for k in ("mode", "exp_name", "lr_scheduler", "n_classes"):
        base.pop(k, None)

    eval_overrides = OmegaConf.create({
        "mode": "eval",
        "seed": 1,
        "load_model_from": "checkpoint",
        "cfg_id": cfg_id,
        "checkpoint_dir": checkpoint_dir,
        "checkpoint_model": f"checkpoint_epoch_{epoch}.pth",
        "pca_labels": pca_labels,
        "pca_n_classes": pca_n_classes,
        "neural_dataset": "things-behavior",
        "analysis": "rsa",
        "compare_method": "spearman",
        "bootstrap": True,
        "log_expdata": True,
        "model_class": "standard_model",
        "model_name": "ResNet50",
        "reconstruct_from_pcs": False,
        "pca_k": 1,
        "region": "N/A",
        "subject_idx": "N/A",
        "batchsize": 64,
        "num_workers": 16,
        "data_augment": False,
        "verbose": False,
    })

    cfg = OmegaConf.merge(base, eval_overrides)
    cfg.return_nodes = list(TORCHVISION_RETURN_NODES["ResNet50"])
    return cfg


def run_things_eval(cfg, model, neural_data_raw, stimuli):
    """Run the THINGS-behavior eval pipeline (mirrors evals.py exactly).

    Steps: SRP extraction -> concept average -> 80/20 split ->
    layer selection -> re-extract best layer without SRP -> bootstrap.
    """
    transform = get_transform(ds_stats="imgnet")
    dl = _make_loader(stimuli, transform, batch=cfg.batchsize, workers=cfg.num_workers)

    # Extract activations (with SRP)
    acts, ids = get_activations(model, dl, DEVICE)

    # Merge train/test, average per concept
    all_concepts = prepare_concept_alignment(cfg, acts, neural_data_raw, ids)
    del acts, ids
    torch.cuda.empty_cache()

    # Fixed 80/20 concept-level split (seed=42, matching production pipeline)
    rng = np.random.RandomState(42)
    n_concepts = all_concepts.neural.size(0)
    perm = rng.permutation(n_concepts)
    n_sel = int(n_concepts * 0.2)
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
            all_concepts.stimulus_ids[i]: all_concepts.concept_image_ids[
                all_concepts.stimulus_ids[i]
            ]
            for i in eval_idx
        },
    )
    del all_concepts

    rprint(
        f"  {n_sel} selection concepts, {len(eval_idx)} evaluation concepts",
        style="success",
    )

    # Build reverse map for re-extraction (image_id -> concepts)
    _img_to_concepts = {}
    for concept in evaluation.stimulus_ids:
        for img_id in evaluation.concept_image_ids[concept]:
            _img_to_concepts.setdefault(str(img_id), []).append(concept)

    def re_extract_fn(layer, sids=None):
        """Re-extract a single layer without SRP, streaming concept-average."""
        model.eval()
        concept_sums, concept_counts = {}, {}
        dl_re = _make_loader(stimuli, transform, batch=cfg.batchsize, workers=cfg.num_workers)
        with torch.no_grad():
            for imgs, keys in dl_re:
                feats = model(imgs.to(DEVICE))
                out = feats[layer].reshape(feats[layer].size(0), -1).cpu().float()
                if cfg.get("reconstruct_from_pcs"):
                    out = reconstruct_from_pcs({layer: out}, cfg.pca_k)[layer]
                for i, key in enumerate(keys):
                    for concept in _img_to_concepts.get(str(key), []):
                        if concept not in concept_sums:
                            concept_sums[concept] = torch.zeros(out.size(1))
                            concept_counts[concept] = 0
                        concept_sums[concept] += out[i]
                        concept_counts[concept] += 1
        avgs = []
        for concept in evaluation.stimulus_ids:
            if concept in concept_sums and concept_counts[concept] > 0:
                avgs.append(concept_sums[concept] / concept_counts[concept])
            else:
                avgs.append(torch.zeros(next(iter(concept_sums.values())).size(0)))
        rprint(
            f"  Re-extracted {layer}: streaming concept-average ({len(avgs)} concepts)",
            style="success",
        )
        return torch.stack(avgs), evaluation.stimulus_ids

    # Layer selection + scoring + bootstrap
    alignment_scores = compute_traintest_alignment(
        cfg, selection, evaluation, verbose=False, re_extract_fn=re_extract_fn
    )

    results = pd.DataFrame(alignment_scores)
    save_results(results, cfg)

    score = results.iloc[0]["score"]
    layer = results.iloc[0]["layer"]
    ci_low = results.iloc[0].get("ci_low")
    ci_high = results.iloc[0].get("ci_high")
    return score, layer, ci_low, ci_high


def main():
    parser = argparse.ArgumentParser(
        description="BN recalibration + THINGS eval (bootstrap) for all ResNet50 models"
    )
    parser.add_argument(
        "--cfg_ids", type=int, nargs="+", default=None,
        help="Specific cfg_ids to evaluate (default: all)",
    )
    parser.add_argument(
        "--n_batches", type=int, default=2000,
        help="Calibration batches of 256 images each (default: 2000 = ~512K images)",
    )
    args = parser.parse_args()

    cfg_ids = args.cfg_ids or sorted(MODEL_CONFIGS.keys())

    # Load THINGS data once (shared across all models)
    print("Loading THINGS data...")
    neural_data_raw, stimuli = load_things_data()
    print(f"  Loaded {len(stimuli)} stimuli\n")

    summary = []
    for cfg_id in cfg_ids:
        if cfg_id not in MODEL_CONFIGS:
            print(f"  Unknown cfg_id={cfg_id}, skipping")
            continue

        ckpt_dir, pca, n_cls, epoch = MODEL_CONFIGS[cfg_id]

        print(f"{'=' * 60}")
        print(f"  cfg{cfg_id} (epoch {epoch}) — BN recalibration + THINGS eval")
        print(f"{'=' * 60}")

        cfg = build_eval_cfg(cfg_id, ckpt_dir, pca, n_cls, epoch)
        model = load_model(cfg, DEVICE, verbose=False)
        model = configure_feature_extractor(cfg, model, verbose=False)

        print("  Recalibrating BN...")
        t0 = time.time()
        recalibrate_bn(model, pca, n_cls, n_batches=args.n_batches)
        cal_time = time.time() - t0
        print(f"    Calibration took {cal_time:.0f}s")

        print("  Running THINGS eval (bootstrap=True)...")
        t0 = time.time()
        score, layer, ci_low, ci_high = run_things_eval(
            cfg, model, neural_data_raw, stimuli
        )
        eval_time = time.time() - t0

        ci_str = ""
        if ci_low is not None and ci_high is not None:
            ci_str = f" [{ci_low:.4f}, {ci_high:.4f}]"
        print(f"\n  Result: {score:.4f}{ci_str}  layer={layer}  ({eval_time:.0f}s)")
        summary.append({
            "cfg_id": cfg_id, "epoch": epoch, "score": score,
            "ci_low": ci_low, "ci_high": ci_high, "layer": layer,
        })

        del model
        torch.cuda.empty_cache()

    # Final summary
    if len(summary) > 1:
        print(f"\n\n{'=' * 60}")
        print("SUMMARY — BN-Recalibrated THINGS Scores (saved to results.db)")
        print(f"{'=' * 60}")
        print(f"{'cfg_id':>6} | {'epoch':>5} | {'score':>8} | {'CI low':>8} | {'CI high':>8} | {'layer':>8}")
        print("-" * 62)
        for r in summary:
            ci_l = f"{r['ci_low']:.4f}" if r['ci_low'] is not None else "   N/A"
            ci_h = f"{r['ci_high']:.4f}" if r['ci_high'] is not None else "   N/A"
            print(f"{r['cfg_id']:>6} | {r['epoch']:>5} | {r['score']:>8.4f} | {ci_l:>8} | {ci_h:>8} | {r['layer']:>8}")


if __name__ == "__main__":
    main()
