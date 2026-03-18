"""
Timed training test on imagenet-mini-50 (50K images) for 5 epochs.
For each architecture, measures per-epoch time, loss, and accuracy,
then extrapolates to full ImageNet (1.28M images) × 90 epochs.

Usage:
    source .venv/bin/activate && python scripts/test_training_timed.py
"""

import sys, os, time
sys.path.insert(0, ".")
from pathlib import Path
assert Path("visreps").is_dir(), "Run from project root"

from dotenv import load_dotenv
load_dotenv()

import torch
import torch.nn as nn
from visreps.utils import load_config, validate_config, setup_optimizer, setup_scheduler
from visreps.models.utils import load_model
from visreps.dataloaders.obj_cls import get_obj_cls_loader
from visreps.utils import calculate_cls_accuracy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_EPOCHS = 5
DATASET = "imagenet-mini-50"
FULL_IMAGENET_IMAGES = 1_281_167  # Full ImageNet train set size

ARCHITECTURES = [
    ("AlexNet",       "configs/train/alexnet.json"),
    ("VGG16",         "configs/train/vgg16.json"),
    ("ResNet18",      "configs/train/resnet18.json"),
    ("ResNet50",      "configs/train/resnet50.json"),
    ("ViTBase",       "configs/train/vit_b_16.json"),
    ("ConvNeXt_Base", "configs/train/convnext_base.json"),
    ("CustomCNN",     "configs/train/base.json"),
]


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"


def test_architecture(name, config_path):
    print(f"\n{'='*70}")
    print(f"  {name} — {DATASET}, {N_EPOCHS} epochs, bs=32")
    print(f"{'='*70}")

    overrides = [
        f"dataset={DATASET}",
        f"num_epochs={N_EPOCHS}",
        "log_interval=1",
        "log_checkpoints=false",
        "use_wandb=false",
        "batchsize=32",
        "num_workers=16",
        "warmup_epochs=1",
    ]
    cfg = load_config(config_path, overrides)
    cfg = validate_config(cfg)

    datasets, loaders = get_obj_cls_loader(cfg)
    n_train = len(datasets["train"])
    n_test = len(datasets["test"])
    num_classes = cfg.pca_n_classes if cfg.pca_labels else datasets["train"].num_classes
    model = load_model(cfg, DEVICE, num_classes=num_classes)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = setup_optimizer(model, cfg)
    scheduler = setup_scheduler(optimizer, cfg)
    label_smoothing = cfg.get("label_smoothing", 0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    use_amp = cfg.get("use_amp", False) and DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=use_amp)

    print(f"  Params: {trainable_params/1e6:.1f}M | Data: {n_train} train, {n_test} test | "
          f"Optimizer: {type(optimizer).__name__} lr={cfg.learning_rate} | AMP: {use_amp}")

    epoch_times = []
    epoch_data = []

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        epoch_start = time.time()
        for images, labels in loaders["train"]:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.autocast(DEVICE.type, enabled=use_amp):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            if cfg.get("grad_clip", 0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            n_batches += 1
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        scheduler.step()

        avg_loss = running_loss / n_batches

        # Evaluate
        model.eval()
        train_top1, _ = calculate_cls_accuracy(loaders["train"], model, DEVICE)
        test_top1, _ = calculate_cls_accuracy(loaders["test"], model, DEVICE)
        lr = optimizer.param_groups[0]["lr"]

        epoch_data.append({
            "epoch": epoch, "loss": avg_loss, "train_acc": train_top1,
            "test_acc": test_top1, "lr": lr, "time": epoch_time
        })
        print(f"  Epoch {epoch}: loss={avg_loss:.4f} | train={train_top1:.1f}% | "
              f"test={test_top1:.1f}% | lr={lr:.6f} | time={epoch_time:.1f}s")

    # ── Extrapolation ───────────────────────────────────────────────────
    avg_epoch_time = sum(epoch_times[1:]) / len(epoch_times[1:])  # Skip first (warmup)
    scale_factor = FULL_IMAGENET_IMAGES / n_train
    full_epoch_time = avg_epoch_time * scale_factor
    target_epochs = cfg.num_epochs  # From original config (before override)

    # Read the original epochs from the config file
    import json
    with open(config_path) as f:
        orig_cfg = json.load(f)
    target_epochs = orig_cfg.get("num_epochs", 90)

    total_time = full_epoch_time * target_epochs

    print(f"\n  --- Extrapolation to full ImageNet ({FULL_IMAGENET_IMAGES:,} images) ---")
    print(f"  Avg epoch time (mini-50): {avg_epoch_time:.1f}s")
    print(f"  Scale factor: {scale_factor:.1f}x")
    print(f"  Est. epoch time (full): {format_time(full_epoch_time)}")
    print(f"  Est. total ({target_epochs} epochs): {format_time(total_time)}")

    # ── Training quality check ──────────────────────────────────────────
    loss_decreased = epoch_data[-1]["loss"] < epoch_data[0]["loss"]
    train_acc_increased = epoch_data[-1]["train_acc"] > epoch_data[0]["train_acc"]
    print(f"\n  Loss: {epoch_data[0]['loss']:.4f} -> {epoch_data[-1]['loss']:.4f} "
          f"({'DECREASING' if loss_decreased else 'WARNING: not decreasing'})")
    print(f"  Train acc: {epoch_data[0]['train_acc']:.1f}% -> {epoch_data[-1]['train_acc']:.1f}% "
          f"({'INCREASING' if train_acc_increased else 'WARNING: not increasing'})")

    return {
        "name": name,
        "params_M": trainable_params / 1e6,
        "avg_epoch_s": avg_epoch_time,
        "est_full_epoch": full_epoch_time,
        "target_epochs": target_epochs,
        "est_total_h": total_time / 3600,
        "final_train_acc": epoch_data[-1]["train_acc"],
        "final_test_acc": epoch_data[-1]["test_acc"],
        "final_loss": epoch_data[-1]["loss"],
        "loss_decreased": loss_decreased,
        "train_acc_increased": train_acc_increased,
    }


def main():
    print("=" * 70)
    print(f"  Timed Training Test — {DATASET} — {N_EPOCHS} epochs — bs=32")
    print(f"  Device: {DEVICE} ({torch.cuda.get_device_name() if DEVICE.type == 'cuda' else 'CPU'})")
    print("=" * 70)

    results = []
    for name, config_path in ARCHITECTURES:
        try:
            r = test_architecture(name, config_path)
            results.append(r)
        except Exception as e:
            print(f"\n  {name}: FAILED — {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()

    # ── Summary table ───────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  SUMMARY — Full ImageNet Training Time Estimates")
    print(f"{'='*70}")
    print(f"  {'Model':<16} {'Params':>8} {'Ep(s)':>7} {'Full Ep':>9} {'Epochs':>6} {'Total':>10} {'Loss':>6} {'TrAcc':>6} {'Learn?':>6}")
    print(f"  {'-'*16} {'-'*8} {'-'*7} {'-'*9} {'-'*6} {'-'*10} {'-'*6} {'-'*6} {'-'*6}")

    for r in results:
        learn = "YES" if (r["loss_decreased"] and r["train_acc_increased"]) else "WARN"
        print(f"  {r['name']:<16} {r['params_M']:>7.1f}M {r['avg_epoch_s']:>6.1f}s "
              f"{format_time(r['est_full_epoch']):>9} {r['target_epochs']:>6} "
              f"{format_time(r['est_total_h']*3600):>10} {r['final_loss']:>6.3f} "
              f"{r['final_train_acc']:>5.1f}% {learn:>6}")

    print(f"\n  Note: Times exclude evaluation. Actual training ~10-15% longer with periodic eval.")


if __name__ == "__main__":
    main()
