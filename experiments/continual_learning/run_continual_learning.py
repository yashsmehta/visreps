"""
Proof-of-principle: Does coarse-grain pretraining help continual learning?

Compares three conditions using Nearest Class Mean (NCM) classification:
  1. from_scratch:         Random init backbone
  2. coarse_pretrained_32: Backbone from K=32 coarse-grained training
  3. fine_pretrained_1000:  Backbone from K=1000 standard ImageNet training

Protocol (class-incremental, NCM):
  - 5 steps, 200 new ImageNet classes per step (total 1000)
  - Backbone is FROZEN throughout — no training at all
  - At each step: compute mean feature vector per new class from training data
  - Classify test images by nearest class mean (cosine similarity)
  - Evaluate: accuracy on all seen classes, current step, and old classes

Usage:
  cd /home/ymehta3/research/VisionAI/visreps
  python experiments/continual_learning/run_continual_learning.py
"""

import os
import sys
import json
import time
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, PROJECT_ROOT)
from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from visreps.dataloaders.obj_cls import ImageNetDataset, get_transform, create_collate_fn
from visreps.models.custom_model import CustomCNN

# ─── Configuration ───────────────────────────────────────────────────────────────

N_STEPS = 5
CLASSES_PER_STEP = 200
BATCH_SIZE = 256
NUM_WORKERS = 8
CLASS_ORDER_SEED = 42
FEATURE_LAYER = "classifier.7"  # post-ReLU fc2 (4096-dim)

CHECKPOINT_PATHS = {
    "coarse_pretrained_32": "/data/ymehta3/alexnet_pca/cfg32a/checkpoint_epoch_20.pth",
    "fine_pretrained_1000": "/data/ymehta3/default/cfg1000a/checkpoint_epoch_20.pth",
}

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ─── Dataset ─────────────────────────────────────────────────────────────────────

def build_class_index(dataset):
    """Build mapping: original class label -> list of sample indices."""
    class_to_indices = defaultdict(list)
    for i, (_, label, _) in enumerate(dataset.samples):
        class_to_indices[label].append(i)
    return dict(class_to_indices)


class ClassSubset(Dataset):
    """Subset of a dataset filtered to specific classes (no label remapping)."""

    def __init__(self, base_dataset, class_set, class_to_indices):
        self.base_dataset = base_dataset
        self.indices = []
        for cls in class_set:
            self.indices.extend(class_to_indices.get(cls, []))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]


def get_class_order(n_classes=1000, seed=42):
    rng = np.random.RandomState(seed)
    return rng.permutation(n_classes).tolist()


# ─── Feature Extraction ─────────────────────────────────────────────────────────

def get_feature_extractor(model, layer):
    """Hook-based feature extractor for a single layer."""
    from torchvision.models.feature_extraction import create_feature_extractor
    return create_feature_extractor(model, return_nodes={layer: layer})


@torch.no_grad()
def extract_features(model, loader, layer, device):
    """Extract features and original labels from a DataLoader."""
    extractor = get_feature_extractor(model, layer)
    extractor.to(device).eval()

    all_features, all_labels = [], []
    for images, labels in loader:
        feats = extractor(images.to(device, non_blocking=True))[layer]
        if feats.dim() > 2:
            feats = feats.flatten(1)
        all_features.append(feats.cpu())
        all_labels.append(labels)

    return torch.cat(all_features), torch.cat(all_labels)


# ─── NCM Classifier ─────────────────────────────────────────────────────────────

def compute_class_means(features, labels, class_set):
    """Compute mean feature vector per class. Returns dict: class -> mean vector."""
    means = {}
    for cls in class_set:
        mask = labels == cls
        means[cls] = features[mask].mean(dim=0)
    return means


def ncm_classify(features, class_means):
    """Classify by nearest class mean (cosine similarity). Returns predicted labels."""
    classes = torch.tensor(sorted(class_means.keys()))
    mean_matrix = torch.stack([class_means[c.item()] for c in classes])

    features_norm = torch.nn.functional.normalize(features, dim=1)
    means_norm = torch.nn.functional.normalize(mean_matrix, dim=1)

    pred_indices = (features_norm @ means_norm.T).argmax(dim=1)
    return classes[pred_indices]


def ncm_accuracy(features, labels, class_means, class_set):
    """Top-1 accuracy on a subset of classes."""
    if not class_set:
        return 0.0
    class_tensor = torch.tensor(sorted(class_set))
    mask = (labels.unsqueeze(1) == class_tensor).any(dim=1)
    if mask.sum() == 0:
        return 0.0
    preds = ncm_classify(features[mask], class_means)
    return 100.0 * (preds == labels[mask]).float().mean().item()


# ─── Model Loading ───────────────────────────────────────────────────────────────

def load_pretrained_backbone(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint["model"].to(device)


def create_fresh_model(device):
    return CustomCNN(num_classes=200).to(device)


# ─── Main Experiment Loop ────────────────────────────────────────────────────────

def run_condition(name, model, device, train_ds, test_ds, train_idx, class_order):
    """Run NCM class-incremental learning for one condition."""
    print(f"\n{'='*65}")
    print(f"  CONDITION: {name}")
    print(f"{'='*65}")

    model.eval()
    class_means = {}  # accumulated across steps
    results = {"condition": name, "steps": [], "all_seen_acc": [], "current_acc": [], "old_acc": []}

    # Extract ALL test features once (backbone is frozen, so they never change)
    print("  Extracting test features...")
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
        pin_memory=True, collate_fn=create_collate_fn(),
    )
    test_feats, test_labels = extract_features(model, test_loader, FEATURE_LAYER, device)

    for step in range(N_STEPS):
        t0 = time.time()
        step_classes = set(class_order[step * CLASSES_PER_STEP : (step + 1) * CLASSES_PER_STEP])
        all_seen = set(class_order[: (step + 1) * CLASSES_PER_STEP])
        old_classes = all_seen - step_classes

        print(f"\n  Step {step+1}/{N_STEPS}: +{CLASSES_PER_STEP} classes -> {len(all_seen)} total")

        # Extract train features for new classes only
        train_loader = DataLoader(
            ClassSubset(train_ds, step_classes, train_idx),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
            pin_memory=True, collate_fn=create_collate_fn(),
        )
        train_feats, train_labels = extract_features(model, train_loader, FEATURE_LAYER, device)

        # Compute means for new classes and add to accumulator
        new_means = compute_class_means(train_feats, train_labels, step_classes)
        class_means.update(new_means)

        # Evaluate
        acc_all = ncm_accuracy(test_feats, test_labels, class_means, all_seen)
        acc_cur = ncm_accuracy(test_feats, test_labels, class_means, step_classes)
        acc_old = ncm_accuracy(test_feats, test_labels, class_means, old_classes)
        dt = time.time() - t0

        print(f"  -> all_seen={acc_all:.1f}%  current={acc_cur:.1f}%  old={acc_old:.1f}%  ({dt:.0f}s)")

        results["steps"].append(step + 1)
        results["all_seen_acc"].append(round(acc_all, 2))
        results["current_acc"].append(round(acc_cur, 2))
        results["old_acc"].append(round(acc_old, 2))

    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.manual_seed(1)
    np.random.seed(1)

    imgnet_path = os.environ["IMAGENET_DATA_DIR"]
    transform = get_transform(ds_stats="imgnet", data_augment=False, image_size=224)

    print("Loading ImageNet splits...")
    train_ds = ImageNetDataset(imgnet_path, split="train", transform=transform)
    test_ds = ImageNetDataset(imgnet_path, split="test", transform=transform)
    print(f"  train={len(train_ds)}  test={len(test_ds)}")

    print("Building class indices...")
    train_idx = build_class_index(train_ds)

    class_order = get_class_order(1000, CLASS_ORDER_SEED)
    all_results = []

    # Condition 1: From scratch (random features)
    model = create_fresh_model(device)
    results = run_condition("from_scratch", model, device, train_ds, test_ds, train_idx, class_order)
    all_results.append(results)
    del model; torch.cuda.empty_cache()

    # Condition 2: Coarse-pretrained (K=32)
    print("\nLoading coarse-pretrained checkpoint (K=32)...")
    model = load_pretrained_backbone(CHECKPOINT_PATHS["coarse_pretrained_32"], device)
    results = run_condition("coarse_pretrained_32", model, device, train_ds, test_ds, train_idx, class_order)
    all_results.append(results)
    del model; torch.cuda.empty_cache()

    # Condition 3: Fine-pretrained (K=1000)
    print("\nLoading fine-pretrained checkpoint (K=1000)...")
    model = load_pretrained_backbone(CHECKPOINT_PATHS["fine_pretrained_1000"], device)
    results = run_condition("fine_pretrained_1000", model, device, train_ds, test_ds, train_idx, class_order)
    all_results.append(results)
    del model; torch.cuda.empty_cache()

    # Save & plot
    results_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")
    plot_results(all_results)


# ─── Plotting ────────────────────────────────────────────────────────────────────

def plot_results(all_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    COLORS = {
        "from_scratch": "#888888",
        "coarse_pretrained_32": "#2196F3",
        "fine_pretrained_1000": "#FF9800",
    }
    LABELS = {
        "from_scratch": "From Scratch",
        "coarse_pretrained_32": "Coarse (K=32)",
        "fine_pretrained_1000": "Fine (K=1000)",
    }

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    metrics = [("all_seen_acc", "All Seen Classes"), ("current_acc", "Current Step"), ("old_acc", "Old Classes")]

    for ax, (key, title) in zip(axes, metrics):
        for r in all_results:
            name = r["condition"]
            ax.plot(r["steps"], r[key], "o-", color=COLORS[name], label=LABELS[name], linewidth=2, markersize=5)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Step")
        ax.set_ylabel("Accuracy (%)")
        ax.set_xticks(range(1, N_STEPS + 1))
        ax.grid(True, alpha=0.2)
        ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "continual_learning.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {path}")
    plt.close()


if __name__ == "__main__":
    main()
