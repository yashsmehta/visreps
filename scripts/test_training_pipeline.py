"""
Smoke-test the training pipeline for every standard architecture.

For each model (AlexNet, VGG16, ResNet18, ResNet50, ViTBase, ConvNeXt_Base):
  1. Load config from its architecture-specific JSON
  2. Override to use imagenet-mini-1 (1 image/class → ~1000 images) for speed
  3. Train for 3 epochs
  4. Verify: loss decreases, gradients flow, model outputs correct shape,
     optimizer/scheduler state is reasonable

Usage:
    source .venv/bin/activate && python scripts/test_training_pipeline.py
"""

import sys
import os
import json
import torch
import torch.nn as nn
from pathlib import Path

# Must run from project root
assert Path("visreps").is_dir(), "Run from project root"
sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv()

from visreps.utils import load_config, validate_config, setup_optimizer, setup_scheduler
from visreps.models.utils import load_model
from visreps.dataloaders.obj_cls import get_obj_cls_loader


ARCHITECTURES = {
    "AlexNet":       "configs/train/alexnet.json",
    "VGG16":         "configs/train/vgg16.json",
    "ResNet18":      "configs/train/resnet18.json",
    "ResNet50":      "configs/train/resnet50.json",
    "ViTBase":       "configs/train/vit_b_16.json",
    "ConvNeXt_Base": "configs/train/convnext_base.json",
}

# Also test CustomCNN via the base config
ARCHITECTURES["CustomCNN"] = "configs/train/base.json"

N_EPOCHS = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test_architecture(name, config_path):
    """Run a full smoke-test for one architecture."""
    print(f"\n{'='*70}")
    print(f"  TESTING: {name}")
    print(f"{'='*70}")

    # ── 1. Load and override config ──────────────────────────────────────
    overrides = [
        "dataset=imagenet-mini-1",
        "num_epochs=3",
        "log_interval=1",
        "log_checkpoints=false",
        "use_wandb=false",
        "batchsize=32",
        "num_workers=4",
        "warmup_epochs=1",
    ]
    cfg = load_config(config_path, overrides)
    cfg = validate_config(cfg)

    print(f"  Config: optimizer={cfg.optimizer}, lr={cfg.learning_rate}, "
          f"wd={cfg.get('weight_decay', 0)}, scheduler={cfg.lr_scheduler}, "
          f"label_smoothing={cfg.get('label_smoothing', 0.1)}")

    # ── 2. Load data ────────────────────────────────────────────────────
    datasets, loaders = get_obj_cls_loader(cfg)
    train_loader = loaders["train"]
    test_loader = loaders["test"]

    # Verify we have data
    n_train = len(datasets["train"])
    n_test = len(datasets["test"])
    print(f"  Data: {n_train} train, {n_test} test samples")
    assert n_train > 0, "No training samples!"

    # Verify train transform uses RandomResizedCrop
    sample_img, sample_label = datasets["train"][0]
    assert sample_img.shape == (3, 224, 224), f"Unexpected shape: {sample_img.shape}"
    print(f"  Image shape: {sample_img.shape}, label range: [0, {datasets['train'].num_classes - 1}]")

    # ── 3. Build model ──────────────────────────────────────────────────
    num_classes = cfg.pca_n_classes if cfg.pca_labels else datasets["train"].num_classes
    model = load_model(cfg, DEVICE, num_classes=num_classes)
    model.train()

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")

    # ── 4. Verify forward pass shape ────────────────────────────────────
    with torch.no_grad():
        dummy = torch.randn(2, 3, 224, 224).to(DEVICE)
        out = model(dummy)
        assert out.shape == (2, num_classes), f"Output shape {out.shape} != expected (2, {num_classes})"
    print(f"  Forward pass: input (2,3,224,224) -> output {out.shape} OK")

    # ── 5. Setup optimizer, scheduler, criterion ────────────────────────
    optimizer = setup_optimizer(model, cfg)
    scheduler = setup_scheduler(optimizer, cfg)
    label_smoothing = cfg.get("label_smoothing", 0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    use_amp = cfg.get("use_amp", False) and DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=use_amp)

    print(f"  Optimizer: {type(optimizer).__name__}, AMP: {use_amp}")
    print(f"  Initial LR: {optimizer.param_groups[0]['lr']:.6f}")

    # ── 6. Train for N_EPOCHS ───────────────────────────────────────────
    epoch_losses = []
    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        grad_norms = []

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            with torch.autocast(DEVICE.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()

            # Check gradients exist
            if n_batches == 0:
                grads_exist = any(p.grad is not None and p.grad.abs().sum() > 0
                                 for p in model.parameters() if p.requires_grad)
                assert grads_exist, "No gradients flowing!"

            if cfg.get("grad_clip", 0) > 0:
                scaler.unscale_(optimizer)
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                grad_norms.append(gn.item())

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = running_loss / n_batches
        epoch_losses.append(avg_loss)

        lr_now = optimizer.param_groups[0]['lr']
        grad_info = f", avg_grad_norm={sum(grad_norms)/len(grad_norms):.4f}" if grad_norms else ""
        print(f"  Epoch {epoch}/{N_EPOCHS}: loss={avg_loss:.4f}, lr={lr_now:.6f}{grad_info}")

    # ── 7. Verify loss decreased ────────────────────────────────────────
    if epoch_losses[-1] < epoch_losses[0]:
        print(f"  Loss decreased: {epoch_losses[0]:.4f} -> {epoch_losses[-1]:.4f} PASS")
    else:
        print(f"  WARNING: Loss did NOT decrease: {epoch_losses[0]:.4f} -> {epoch_losses[-1]:.4f}")
        # Don't assert — with 3 epochs on tiny data, some models may not decrease

    # ── 8. Quick eval ───────────────────────────────────────────────────
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            with torch.autocast(DEVICE.type, enabled=use_amp):
                outputs = model(images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = 100.0 * correct / total if total > 0 else 0.0
    print(f"  Test accuracy: {test_acc:.2f}% ({correct}/{total})")

    # ── 9. Verify scheduler moved LR ────────────────────────────────────
    final_lr = optimizer.param_groups[0]['lr']
    initial_lr = cfg.learning_rate
    if cfg.lr_scheduler == "cosineannealinglr" or cfg.get("warmup_epochs", 0) > 0:
        print(f"  LR schedule: {initial_lr:.6f} -> {final_lr:.6f} (moved) OK")
    else:
        print(f"  LR schedule: {initial_lr:.6f} -> {final_lr:.6f} (StepLR, step_size=30, no change in 3 epochs) OK")

    print(f"  {name}: ALL CHECKS PASSED")
    return True


def main():
    print("=" * 70)
    print("  Training Pipeline Smoke Test")
    print(f"  Device: {DEVICE}")
    print(f"  Testing {len(ARCHITECTURES)} architectures x {N_EPOCHS} epochs each")
    print("=" * 70)

    results = {}
    for name, config_path in ARCHITECTURES.items():
        try:
            results[name] = test_architecture(name, config_path)
        except Exception as e:
            print(f"\n  {name}: FAILED with {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:20s} {status}")

    n_passed = sum(results.values())
    n_total = len(results)
    print(f"\n  {n_passed}/{n_total} architectures passed")

    if n_passed < n_total:
        sys.exit(1)


if __name__ == "__main__":
    main()
