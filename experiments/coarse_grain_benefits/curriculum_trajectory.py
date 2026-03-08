"""
Curriculum Trajectory: Track brain alignment during granularity fine-tuning.

Runs one direction at a time (coarse->fine or fine->coarse). At each sub-epoch
checkpoint (every 1/4 epoch), measures:
  1. Brain alignment (RSA with NSD, averaged across 8 subjects)
  2. Representation drift (RDM similarity to source model)
  3. Classification accuracy (on ImageNet val subset)

The best layer per NSD region is queried from results.db at startup, then
tracked throughout fine-tuning to show how alignment evolves.

Usage:
  python curriculum_trajectory.py --source_cfg_id 32  --target_cfg_id 1000
  python curriculum_trajectory.py --source_cfg_id 1000 --target_cfg_id 32
"""

import os
import sys
import argparse
import time
import sqlite3

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import transforms

from visreps.models.utils import FeatureExtractor
from visreps.analysis.rsa import compute_rdm, compute_rdm_correlation
from visreps.dataloaders.neural import load_all_nsd_data, _make_loader
from visreps.dataloaders.obj_cls import get_obj_cls_loader

from utils import get_device, load_coarse_model, OUTPUT_DIR
from curriculum_finetuning import (
    COARSE_CHECKPOINT_DIR, COARSE_LABELS_FOLDER,
    FINE_CHECKPOINT_DIR, replace_classifier_head,
)

# ─────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────
NSD_REGIONS = ["early visual stream", "ventral visual stream"]
REGION_SHORT = {
    "early visual stream": "early",
    "ventral visual stream": "ventral",
}


# ─────────────────────────────────────────────────────────
# Database query
# ─────────────────────────────────────────────────────────
def query_best_layers(cfg_id):
    """Query results.db for the best RSA layer per NSD region."""
    db_path = os.path.join(PROJECT_ROOT, "results.db")
    conn = sqlite3.connect(db_path)
    best = {}
    for region in NSD_REGIONS:
        df = pd.read_sql(
            "SELECT layer, AVG(score) as avg_score FROM results "
            "WHERE neural_dataset='nsd' AND compare_method='spearman' "
            "AND analysis='rsa' AND cfg_id=? AND region=? "
            "GROUP BY layer ORDER BY avg_score DESC LIMIT 1",
            conn, params=(cfg_id, region),
        )
        best[region] = df.iloc[0]["layer"] if len(df) else "fc1_pre"
    conn.close()
    return best


# ─────────────────────────────────────────────────────────
# NSD probe data
# ─────────────────────────────────────────────────────────
def load_nsd_probe():
    """Load NSD shared test data: per-subject neural RDMs and stimulus loader.

    Returns:
        neural_rdms: {region: [rdm_subj0, rdm_subj1, ...]} — precomputed once
        nsd_loader: DataLoader over ~1000 shared test stimuli
    """
    print("Loading NSD probe data...")
    nsd = load_all_nsd_data({}, subjects=list(range(8)), regions=NSD_REGIONS)
    # IMPORTANT: sort as strings (lexicographic) to match _StimuliDataset ordering.
    # shared_test_ids is sorted numerically, but the DataLoader sorts keys as strings.
    test_ids = sorted(nsd["shared_test_ids"])
    n_subjects = len(nsd["subjects"])

    # Per-subject neural RDMs (computed once, reused at every checkpoint)
    neural_rdms = {}
    for region in NSD_REGIONS:
        neural_rdms[region] = []
        for subj in nsd["subjects"]:
            responses = np.stack([
                nsd["neural"][region][subj]["test"][sid] for sid in test_ids
            ])
            rdm = compute_rdm(torch.tensor(responses, dtype=torch.float32))
            neural_rdms[region].append(rdm)

    # Load shared test images into memory (~540 MB for ~1000 images)
    test_stimuli = {sid: nsd["stimuli"][sid] for sid in test_ids}
    nsd_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    nsd_loader = _make_loader(test_stimuli, nsd_transform, batch=128, workers=4)

    print(f"  {len(test_ids)} shared test stimuli, {n_subjects} subjects")
    return neural_rdms, nsd_loader


# ─────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────
def extract_at_layers(model, loader, layers, device):
    """Extract features at specific layers (e.g., 'fc1_pre', 'conv4_post').

    Creates a temporary FeatureExtractor, extracts, and removes hooks.
    """
    # Derive base layer names (e.g., 'fc1' from 'fc1_pre')
    base_layers = set()
    for layer in layers:
        base = layer.rsplit("_", 1)[0] if ("_pre" in layer or "_post" in layer) else layer
        base_layers.add(base)

    extractor = FeatureExtractor(
        model, return_nodes={l: l for l in base_layers},
        post_relu=True, extract_pre_and_post=True,
    )
    extractor.to(device).eval()

    feats = {l: [] for l in layers}
    with torch.no_grad():
        for images, _ in loader:
            out = extractor(images.to(device))
            for layer in layers:
                f = out[layer]
                if f.dim() > 2:
                    f = f.view(f.size(0), -1)
                feats[layer].append(f.cpu())

    for h in extractor.handles:
        h.remove()

    return {l: torch.cat(feats[l], 0) for l in layers}


# ─────────────────────────────────────────────────────────
# Checkpoint evaluation
# ─────────────────────────────────────────────────────────
def evaluate_checkpoint(model, nsd_loader, neural_rdms, source_rdms,
                        best_layers, device):
    """Compute RSA (brain alignment) and drift (similarity to source) at a checkpoint."""
    was_training = model.training
    model.eval()

    layers_needed = list(set(best_layers.values()))
    features = extract_at_layers(model, nsd_loader, layers_needed, device)

    metrics = {}
    for region in NSD_REGIONS:
        short = REGION_SHORT[region]
        layer = best_layers[region]
        model_rdm = compute_rdm(features[layer])

        # RSA: Spearman correlation with each subject's neural RDM, then average
        scores = [
            compute_rdm_correlation(model_rdm, nrdm, correlation="Spearman")
            for nrdm in neural_rdms[region]
        ]
        metrics[f"rsa_{short}"] = np.mean(scores)

        # Drift: RDM correlation with source model (1.0 = identical, decreases with change)
        metrics[f"drift_{short}"] = compute_rdm_correlation(
            model_rdm, source_rdms[layer], correlation="Spearman"
        )

    if was_training:
        model.train()
    return metrics


@torch.no_grad()
def quick_accuracy(model, loader, device, n_batches=20):
    """Validation accuracy on first n_batches (~5K images)."""
    was_training = model.training
    model.eval()
    correct = total = 0
    for i, (images, labels) in enumerate(loader):
        if i >= n_batches:
            break
        out = model(images.to(device))
        correct += out.argmax(1).eq(labels.to(device)).sum().item()
        total += labels.size(0)
    if was_training:
        model.train()
    return 100.0 * correct / total if total else 0.0


# ─────────────────────────────────────────────────────────
# Main trajectory
# ─────────────────────────────────────────────────────────
def run_trajectory(
    source_cfg_id=32,
    target_cfg_id=1000,
    seed=1,
    num_epochs=5,
    learning_rate=5e-4,
    batch_size=256,
    num_workers=8,
):
    device = get_device()
    output_dir = os.path.join(OUTPUT_DIR, "curriculum_trajectory")
    os.makedirs(output_dir, exist_ok=True)

    seed_letter = chr(ord("a") + seed - 1)
    exp_name = f"cfg{source_cfg_id}_to_{target_cfg_id}_{seed_letter}"

    print(f"\n{'='*60}")
    print(f"Curriculum Trajectory: {source_cfg_id}-way -> {target_cfg_id}-way")
    print(f"Seed: {seed} ({seed_letter}) | LR: {learning_rate} | Epochs: {num_epochs}")
    print(f"{'='*60}")

    # ── 1. Best layers from DB ──────────────────────────────
    best_layers = query_best_layers(source_cfg_id)
    print("\nBest layers (from results.db):")
    for region, layer in best_layers.items():
        print(f"  {REGION_SHORT[region]}: {layer}")

    # ── 2. Load source model ────────────────────────────────
    print(f"\nLoading {source_cfg_id}-way source model...")
    ckpt_dir = FINE_CHECKPOINT_DIR if source_cfg_id == 1000 else COARSE_CHECKPOINT_DIR
    model = load_coarse_model(source_cfg_id, seed, ckpt_dir, device)

    # ── 3. Load NSD probe data ──────────────────────────────
    neural_rdms, nsd_loader = load_nsd_probe()

    # ── 4. Source model RDMs (drift reference) ──────────────
    print("Computing source model RDMs...")
    layers_needed = list(set(best_layers.values()))
    src_features = extract_at_layers(model, nsd_loader, layers_needed, device)
    source_rdms = {layer: compute_rdm(src_features[layer]) for layer in layers_needed}
    del src_features

    # ── 5. Replace classifier head ──────────────────────────
    model = replace_classifier_head(model, source_cfg_id, target_cfg_id)
    model = model.to(device)

    # ── 6. Load ImageNet data ───────────────────────────────
    print(f"Loading ImageNet ({target_cfg_id}-way)...")
    loader_cfg = {
        "dataset": "imagenet",
        "batchsize": batch_size,
        "num_workers": num_workers,
        "pca_labels": target_cfg_id != 1000,
        "pca_n_classes": target_cfg_id,
        "pca_labels_folder": COARSE_LABELS_FOLDER,
        "data_augment": True,
    }
    _, loaders = get_obj_cls_loader(
        loader_cfg, shuffle=True, preprocess=True, train_test_split=True
    )

    # ── 7. Optimizer + AMP ──────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    scaler = torch.amp.GradScaler("cuda")

    # ── 8. Sub-epoch checkpoint setup ───────────────────────
    n_batches = len(loaders["train"])
    ckpt_every = max(1, n_batches // 4)
    print(f"\n{n_batches} batches/epoch, checkpoint every {ckpt_every} (~1/4 epoch)")

    # ── 9. Initial evaluation (step 0) ──────────────────────
    results = []

    def log_checkpoint(step, epoch_frac, train_loss, metrics, val_acc):
        row = {
            "source_cfg_id": source_cfg_id,
            "target_cfg_id": target_cfg_id,
            "seed": seed,
            "step": step,
            "epoch_frac": epoch_frac,
            "train_loss": train_loss,
            "val_top1": val_acc,
            **metrics,
        }
        for region in NSD_REGIONS:
            row[f"layer_{REGION_SHORT[region]}"] = best_layers[region]
        results.append(row)

    print("\n--- Step 0 (before fine-tuning) ---")
    metrics = evaluate_checkpoint(
        model, nsd_loader, neural_rdms, source_rdms, best_layers, device
    )
    val_acc = quick_accuracy(model, loaders["test"], device)
    log_checkpoint(0, 0.0, None, metrics, val_acc)

    for region in NSD_REGIONS:
        s = REGION_SHORT[region]
        print(f"  RSA {s}: {metrics[f'rsa_{s}']:.4f} | "
              f"Drift {s}: {metrics[f'drift_{s}']:.4f}")
    print(f"  Val Top-1: {val_acc:.2f}%")

    # ── 10. Training loop ───────────────────────────────────
    print(f"\n{'='*60}")
    print("Starting fine-tuning")
    print(f"{'='*60}")

    global_step = 0
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        quarter_loss = 0.0
        quarter_count = 0

        pbar = tqdm(loaders["train"], desc=f"Epoch {epoch}", leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            quarter_loss += loss.item()
            quarter_count += 1
            global_step += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            # Sub-epoch checkpoint
            if (batch_idx + 1) % ckpt_every == 0:
                quarter = (batch_idx + 1) // ckpt_every
                epoch_frac = epoch - 1 + quarter / 4
                avg_loss = quarter_loss / quarter_count

                metrics = evaluate_checkpoint(
                    model, nsd_loader, neural_rdms, source_rdms,
                    best_layers, device,
                )
                val_acc = quick_accuracy(model, loaders["test"], device)
                log_checkpoint(global_step, epoch_frac, avg_loss, metrics, val_acc)

                print(f"\n  [{epoch_frac:.2f}] Loss: {avg_loss:.4f} | Val: {val_acc:.2f}%")
                for region in NSD_REGIONS:
                    s = REGION_SHORT[region]
                    print(f"    RSA {s}: {metrics[f'rsa_{s}']:.4f} | "
                          f"Drift {s}: {metrics[f'drift_{s}']:.4f}")

                quarter_loss = 0.0
                quarter_count = 0
                model.train()

    elapsed = time.time() - start_time
    print(f"\nDone! {elapsed / 60:.1f} minutes")

    # Save results
    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, f"{exp_name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    return df


def main():
    p = argparse.ArgumentParser(
        description="Track brain alignment during curriculum fine-tuning"
    )
    p.add_argument("--source_cfg_id", type=int, default=32)
    p.add_argument("--target_cfg_id", type=int, default=1000)
    p.add_argument("--seed", type=int, default=1, choices=[1, 2, 3])
    p.add_argument("--num_epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    args = p.parse_args()

    run_trajectory(
        source_cfg_id=args.source_cfg_id,
        target_cfg_id=args.target_cfg_id,
        seed=args.seed,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
