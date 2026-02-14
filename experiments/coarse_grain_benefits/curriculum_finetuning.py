"""
Curriculum Learning Experiment: Source → Target Granularity Fine-Tuning

Fine-tunes a model trained at one label granularity on a different granularity.
Supports any direction: coarse→fine, fine→coarse, or coarse→coarse.

Examples:
  64-way → 1000-way   (coarse pre-training helps fine-grained?)
  1000-way → 32-way   (does fine-grained knowledge transfer to coarse?)
  32-way → 64-way     (curriculum across coarse levels)

Transfer modes:
- full: Train all layers (standard fine-tuning)
- late_layers: Freeze conv1-4, train conv5 + fc layers (hierarchical transfer)
- fc_only: Freeze all conv layers, train only fc layers
- head_only: Freeze everything except the new classification head

Usage:
  python curriculum_finetuning.py --source_cfg_id 64  --target_cfg_id 1000
  python curriculum_finetuning.py --source_cfg_id 1000 --target_cfg_id 32
  python curriculum_finetuning.py --source_cfg_id 32  --target_cfg_id 64

Output:
- Checkpoints: {output_dir}/cfg{source}_to_{target}_{mode}_{seed}/checkpoint_epoch_{n}.pth
- Metrics CSV: {output_dir}/curriculum_finetuning_metrics.csv
"""

import os
import sys
import argparse
import time
import json

# Setup paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm

from visreps.dataloaders.obj_cls import get_obj_cls_loader
from visreps.utils import calculate_cls_accuracy

from utils import (
    get_device, ensure_output_dir, load_coarse_model,
    get_config_name, OUTPUT_DIR
)


# =============================================================================
# Coarse Model Configuration
# =============================================================================
# Change this to switch between coarse-label families (e.g., "alexnet", "dino").
# Checkpoint dir and PCA labels folder are derived automatically:
#   Checkpoint dir: /data/ymehta3/{COARSE_MODEL}_pca/
#   PCA labels:     pca_labels/pca_labels_{COARSE_MODEL}/

COARSE_MODEL = "alexnet"
COARSE_CHECKPOINT_DIR = f"/data/ymehta3/{COARSE_MODEL}_pca"
COARSE_LABELS_FOLDER = f"pca_labels_{COARSE_MODEL}"

# 1000-way models always live here
FINE_CHECKPOINT_DIR = "/data/ymehta3/default"


# =============================================================================
# Transfer Mode Presets
# =============================================================================
# Each mode specifies which layers are trainable via binary strings:
#   conv: "11111" = all 5 conv layers trainable, "00001" = only conv5
#   fc: "111" = all 3 fc layers trainable (fc1, fc2, head)

TRANSFER_MODES = {
    "full": {
        "conv": "11111",  # All conv layers trainable
        "fc": "111",      # All fc layers trainable
        "description": "Train all layers (standard fine-tuning)"
    },
    "late_layers": {
        "conv": "00001",  # Only conv5 trainable (freeze conv1-4)
        "fc": "111",      # All fc layers trainable
        "description": "Freeze conv1-4, train conv5 + fc (hierarchical transfer)"
    },
    "fc_only": {
        "conv": "00000",  # Freeze all conv layers
        "fc": "111",      # All fc layers trainable
        "description": "Freeze all conv, train only fc layers"
    },
    "head_only": {
        "conv": "00000",  # Freeze all conv layers
        "fc": "001",      # Only classification head trainable
        "description": "Freeze everything except classification head"
    },
}


def set_transfer_mode(model, mode):
    """
    Configure which layers are trainable based on transfer mode.

    Args:
        model: CustomCNN model with _set_trainable_layers method
        mode: One of 'full', 'late_layers', 'fc_only', 'head_only'

    Returns:
        model with appropriate layers frozen
    """
    if mode not in TRANSFER_MODES:
        raise ValueError(f"Unknown transfer mode: {mode}. Choose from {list(TRANSFER_MODES.keys())}")

    config = TRANSFER_MODES[mode]
    trainable_layers = {"conv": config["conv"], "fc": config["fc"]}

    model._set_trainable_layers(trainable_layers)

    print(f"Transfer mode: {mode}")
    print(f"  {config['description']}")
    print(f"  Conv trainable: {config['conv']} | FC trainable: {config['fc']}")

    return model


def replace_classifier_head(model, old_num_classes, new_num_classes=1000):
    """
    Replace the final classifier layer to output new_num_classes.

    For CustomCNN, the classifier is a Sequential with the final Linear at index 8.
    """
    # Get the final linear layer
    final_layer = model.classifier[8]

    if not isinstance(final_layer, nn.Linear):
        raise ValueError(f"Expected final layer to be Linear, got {type(final_layer)}")

    in_features = final_layer.in_features

    if final_layer.out_features != old_num_classes:
        raise ValueError(
            f"Expected {old_num_classes} output classes, got {final_layer.out_features}"
        )

    # Replace with new classifier head
    model.classifier[8] = nn.Linear(in_features, new_num_classes)

    # Initialize the new layer (He initialization)
    nn.init.kaiming_normal_(model.classifier[8].weight, mode='fan_out', nonlinearity='relu')
    nn.init.zeros_(model.classifier[8].bias)

    # Update model's num_classes attribute if it exists
    if hasattr(model, 'num_classes'):
        model.num_classes = new_num_classes

    print(f"Replaced classifier: {old_num_classes} → {new_num_classes} classes")
    return model


def get_imagenet_loaders(target_cfg_id=1000, batch_size=256, num_workers=8):
    """Get ImageNet dataloaders with labels matching target_cfg_id."""
    cfg = {
        "dataset": "imagenet",
        "batchsize": batch_size,
        "num_workers": num_workers,
        "pca_labels": target_cfg_id != 1000,
        "pca_n_classes": target_cfg_id,
        "pca_labels_folder": COARSE_LABELS_FOLDER,
        "data_augment": True,
    }

    datasets, loaders = get_obj_cls_loader(cfg, shuffle=True, preprocess=True, train_test_split=True)
    return datasets, loaders


@torch.no_grad()
def evaluate(model, loader, device):
    """Evaluate model on a dataloader, return top-1 and top-5 accuracy."""
    model.eval()
    return calculate_cls_accuracy(loader, model, device)


def train_epoch(model, loader, criterion, optimizer, device, epoch, use_amp=True):
    """Train for one epoch, return average loss."""
    model.train()
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    total_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return total_loss / len(loader)


def save_checkpoint(checkpoint_dir, epoch, model, optimizer, scheduler, metrics, config):
    """Save checkpoint with model, optimizer state, and metrics."""
    checkpoint = {
        'epoch': epoch,
        'model': model,  # Save entire model (consistent with existing codebase)
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': config,
    }
    path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, path)
    return path


def run_curriculum_finetuning(
    source_cfg_id=64,
    target_cfg_id=1000,
    seed=1,
    num_epochs=10,
    learning_rate=0.002,
    weight_decay=0.0001,
    batch_size=256,
    num_workers=8,
    warmup_epochs=1,
    transfer_mode="full",
    eval_freq=2,
    output_dir=None,
):
    """
    Fine-tune a model from one label granularity to another.

    Args:
        source_cfg_id: Granularity of the pretrained model (e.g., 64, 1000)
        target_cfg_id: Granularity to fine-tune towards (e.g., 1000, 32)
        seed: Training seed (1, 2, or 3)
        num_epochs: Number of fine-tuning epochs
        learning_rate: Learning rate for fine-tuning
        weight_decay: Weight decay for AdamW
        batch_size: Batch size for training
        num_workers: Number of dataloader workers
        warmup_epochs: Number of warmup epochs
        transfer_mode: Which layers to train (full, late_layers, fc_only, head_only)
        eval_freq: Frequency (in epochs) to evaluate accuracy (default: 2)
        output_dir: Directory to save fine-tuned checkpoints
    """
    device = get_device()
    print(f"Device: {device}")

    # Source checkpoint dir depends on whether source is 1000-way or coarse
    source_checkpoint_dir = FINE_CHECKPOINT_DIR if source_cfg_id == 1000 else COARSE_CHECKPOINT_DIR

    if output_dir is None:
        output_dir = os.path.join(OUTPUT_DIR, "curriculum_checkpoints")

    # Create output directory for this experiment
    seed_letter = chr(ord('a') + seed - 1)
    exp_name = f"cfg{source_cfg_id}_to_{target_cfg_id}_{transfer_mode}_{seed_letter}"
    exp_dir = os.path.join(output_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Curriculum Fine-tuning: {source_cfg_id}-way → {target_cfg_id}-way")
    print(f"{'='*60}")
    print(f"Coarse model: {COARSE_MODEL}")
    print(f"Seed: {seed} ({seed_letter})")
    print(f"Transfer mode: {transfer_mode}")
    print(f"Fine-tuning epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print(f"Output directory: {exp_dir}")

    # Load source model
    print(f"\n=== Loading {source_cfg_id}-way model ===")
    model = load_coarse_model(source_cfg_id, seed, source_checkpoint_dir, device)

    # Replace classifier head
    print(f"\n=== Replacing classifier head ===")
    model = replace_classifier_head(model, source_cfg_id, target_cfg_id)
    model = model.to(device)

    # Apply transfer mode (freeze appropriate layers)
    print(f"\n=== Applying transfer mode ===")
    model = set_transfer_mode(model, transfer_mode)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {frozen_params:,}")

    # Load ImageNet data (with PCA labels if target is coarse)
    print(f"\n=== Loading ImageNet ({target_cfg_id}-way labels) ===")
    datasets, loaders = get_imagenet_loaders(target_cfg_id, batch_size, num_workers)
    print(f"Train samples: {len(datasets['train'])}")
    print(f"Val samples: {len(datasets['test'])}")

    # Setup training (only optimize trainable parameters)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    # Cosine annealing scheduler with warmup
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            import math
            progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Save config
    config = {
        'source_cfg_id': source_cfg_id,
        'target_cfg_id': target_cfg_id,
        'coarse_model': COARSE_MODEL,
        'seed': seed,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'weight_decay': weight_decay,
        'batch_size': batch_size,
        'warmup_epochs': warmup_epochs,
        'transfer_mode': transfer_mode,
        'transfer_mode_config': TRANSFER_MODES[transfer_mode],
        'eval_freq': eval_freq,
        'source_checkpoint_dir': source_checkpoint_dir,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'frozen_params': frozen_params,
    }

    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Training loop
    print(f"\n=== Starting Fine-tuning ===")
    results = []
    best_acc = 0

    # Initial evaluation (epoch 0)
    print(f"\nEpoch 0 (before fine-tuning):")
    val_top1, val_top5 = evaluate(model, loaders['test'], device)
    print(f"  Val Top-1: {val_top1:.2f}%  |  Val Top-5: {val_top5:.2f}%")

    results.append({
        'source_cfg_id': source_cfg_id,
        'target_cfg_id': target_cfg_id,
        'seed': seed,
        'transfer_mode': transfer_mode,
        'epoch': 0,
        'train_loss': None,
        'val_top1': val_top1,
        'val_top5': val_top5,
        'learning_rate': learning_rate,
    })

    # Save initial checkpoint
    metrics = {'val_top1': val_top1, 'val_top5': val_top5}
    save_checkpoint(exp_dir, 0, model, optimizer, scheduler, metrics, config)

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(
            model, loaders['train'], criterion, optimizer, device, epoch
        )

        # Update scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        epoch_time = time.time() - epoch_start

        print(f"\nEpoch {epoch}/{num_epochs}:")
        print(f"  Train Loss: {train_loss:.4f}  |  LR: {current_lr:.6f}")
        print(f"  Time: {epoch_time:.1f}s")

        # Evaluate accuracy every eval_freq epochs (or final epoch)
        val_top1, val_top5 = None, None
        if epoch % eval_freq == 0 or epoch == num_epochs:
            val_top1, val_top5 = evaluate(model, loaders['test'], device)
            print(f"  Val Top-1: {val_top1:.2f}%  |  Val Top-5: {val_top5:.2f}%")

            # Track best
            if val_top1 > best_acc:
                best_acc = val_top1
                print(f"  ★ New best!")

            # Save results
            results.append({
                'source_cfg_id': source_cfg_id,
                'target_cfg_id': target_cfg_id,
                'seed': seed,
                'transfer_mode': transfer_mode,
                'epoch': epoch,
                'train_loss': train_loss,
                'val_top1': val_top1,
                'val_top5': val_top5,
                'learning_rate': current_lr,
            })

        # Save checkpoint every epoch
        metrics = {
            'train_loss': train_loss,
            'val_top1': val_top1,
            'val_top5': val_top5,
        }
        save_checkpoint(exp_dir, epoch, model, optimizer, scheduler, metrics, config)

        # ETA
        if epoch == 1:
            eta = (time.time() - start_time) * (num_epochs - 1)
            print(f"  ETA: {eta/60:.1f} minutes")

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"Fine-tuning complete!")
    print(f"Best Val Top-1: {best_acc:.2f}%")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Checkpoints saved to: {exp_dir}")
    print(f"{'='*60}")

    # Save results CSV
    df = pd.DataFrame(results)
    csv_path = os.path.join(exp_dir, 'metrics.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nMetrics saved to: {csv_path}")

    # Also append to global results file
    global_csv = os.path.join(OUTPUT_DIR, 'curriculum_finetuning_all.csv')
    if os.path.exists(global_csv):
        df_global = pd.read_csv(global_csv)
        df_global = pd.concat([df_global, df], ignore_index=True)
    else:
        df_global = df
    df_global.to_csv(global_csv, index=False)

    return df, exp_dir


def main():
    parser = argparse.ArgumentParser(
        description="Curriculum fine-tuning: source → target granularity"
    )
    parser.add_argument(
        "--source_cfg_id", type=int, default=64,
        help="Granularity of pretrained model to load (default: 64)"
    )
    parser.add_argument(
        "--target_cfg_id", type=int, default=1000,
        help="Granularity to fine-tune towards (default: 1000)"
    )
    parser.add_argument(
        "--seed", type=int, default=1, choices=[1, 2, 3],
        help="Training seed (default: 1)"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10,
        help="Number of fine-tuning epochs (default: 10)"
    )
    parser.add_argument(
        "--lr", type=float, default=0.002,
        help="Learning rate (default: 0.002, same as original training)"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0001,
        help="Weight decay (default: 0.0001)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256,
        help="Batch size (default: 256)"
    )
    parser.add_argument(
        "--num_workers", type=int, default=8,
        help="Number of dataloader workers (default: 8)"
    )
    parser.add_argument(
        "--warmup_epochs", type=int, default=1,
        help="Number of warmup epochs (default: 1)"
    )
    parser.add_argument(
        "--transfer_mode", type=str, default="full",
        choices=["full", "late_layers", "fc_only", "head_only"],
        help="Which layers to train: full (all), late_layers (conv5+fc), fc_only, head_only"
    )
    parser.add_argument(
        "--eval_freq", type=int, default=2,
        help="Frequency (in epochs) to evaluate accuracy (default: 2)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Directory to save fine-tuned checkpoints"
    )

    args = parser.parse_args()

    run_curriculum_finetuning(
        source_cfg_id=args.source_cfg_id,
        target_cfg_id=args.target_cfg_id,
        seed=args.seed,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        warmup_epochs=args.warmup_epochs,
        transfer_mode=args.transfer_mode,
        eval_freq=args.eval_freq,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
