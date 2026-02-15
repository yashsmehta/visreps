"""
Few-Shot Learning: Tests if coarse pre-training produces more transferable representations.

Protocol:
1. Load models from checkpoints
2. Use CIFAR-100 as the transfer dataset
3. For k in {1, 5, 10, 20} shots per class:
   - Sample k examples per class, extract features (frozen), train logistic regression
4. Compare across models
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from utils import get_device, ensure_output_dir, get_feature_extractor, load_checkpoint_model


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
MODELS = {
    "AlexNet (1K classes)": "/data/ymehta3/default/cfg1000a/checkpoint_epoch_20.pth",
    "AlexNet (64 classes)": "/data/ymehta3/alexnet_pca/cfg64a/checkpoint_epoch_20.pth",
    "AlexNet (64→1K curriculum)": "experiments/coarse_grain_benefits/results/curriculum_checkpoints/cfg64_to_1000_late_layers_a/checkpoint_epoch_10.pth",
}

LAYER = "fc2"
K_SHOTS = [1, 5, 10, 20]
N_TRIALS = 3


# ─────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────
def get_cifar100_loaders(batch_size=256, num_workers=4):
    """Load CIFAR-100 with transforms for AlexNet (resize to 224x224)."""
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    cifar_root = os.path.join(PROJECT_ROOT, 'data', 'obj_cls', 'cifar-100')
    train_dataset = torchvision.datasets.CIFAR100(
        root=cifar_root, train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR100(
        root=cifar_root, train=False, download=True, transform=transform
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_dataset, test_loader


def sample_k_shot(dataset, k_shots, seed=42):
    """Sample k examples per class from the dataset."""
    np.random.seed(seed)
    targets = np.array(dataset.targets)
    classes = np.unique(targets)

    selected_indices = []
    for cls in classes:
        cls_indices = np.where(targets == cls)[0]
        selected = np.random.choice(cls_indices, min(k_shots, len(cls_indices)), replace=False)
        selected_indices.extend(selected)

    return Subset(dataset, selected_indices)


# ─────────────────────────────────────────────────────────────
# EVALUATION
# ─────────────────────────────────────────────────────────────
def few_shot_trials(test_features, test_labels, train_dataset, extractor, k_shots, n_trials, layer, device):
    """Run k-shot trials with pre-extracted test features."""
    per_trial_accs = []
    for trial in range(n_trials):
        train_subset = sample_k_shot(train_dataset, k_shots, seed=trial)
        train_loader = DataLoader(train_subset, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
        train_features, train_labels = extract_features_with_extractor(extractor, train_loader, layer, device)

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_features)
        test_scaled = scaler.transform(test_features)

        clf = LogisticRegression(max_iter=1000, n_jobs=-1)
        clf.fit(train_scaled, train_labels)
        per_trial_accs.append(clf.score(test_scaled, test_labels))

    return np.mean(per_trial_accs), np.std(per_trial_accs), per_trial_accs


def extract_features_with_extractor(extractor, loader, layer, device):
    """Extract features using pre-built extractor."""
    features_list, labels_list = [], []
    with torch.no_grad():
        for images, labels in loader:
            feats = extractor(images.to(device))[layer]
            if feats.dim() > 2:
                feats = feats.view(feats.size(0), -1)
            features_list.append(feats.cpu().numpy())
            labels_list.append(labels.numpy())
    return np.vstack(features_list), np.concatenate(labels_list)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    device = get_device()
    print(f"Device: {device}")

    print("\n=== Loading CIFAR-100 ===")
    train_dataset, test_loader = get_cifar100_loaders()
    print(f"Train: {len(train_dataset)}, Test: {len(test_loader.dataset)}, Classes: 100")

    results = []

    for name, checkpoint_path in MODELS.items():
        print(f"\n=== {name} ===")

        try:
            model = load_checkpoint_model(checkpoint_path, device)
        except FileNotFoundError as e:
            print(f"Skipping: {e}")
            continue

        # Build extractor once per model
        extractor = get_feature_extractor(model, LAYER).to(device).eval()

        # Extract test features once per model
        print("  Extracting test features...")
        test_features, test_labels = extract_features_with_extractor(extractor, test_loader, LAYER, device)

        for k in K_SHOTS:
            mean_acc, std_acc, per_trial = few_shot_trials(
                test_features, test_labels, train_dataset, extractor, k, N_TRIALS, LAYER, device
            )
            print(f"  k={k}: {mean_acc*100:.2f}% (+/- {std_acc*100:.2f}%)")

            for trial_idx, trial_acc in enumerate(per_trial):
                results.append({
                    'model_name': name,
                    'layer': LAYER,
                    'k_shots': k,
                    'trial': trial_idx,
                    'accuracy': trial_acc,
                })

        del model
        torch.cuda.empty_cache()

    # Save
    output_file = os.path.join(ensure_output_dir(), "few_shot_learning.csv")
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")

    # Summary
    print("\n=== Summary ===")
    summary = df.groupby(['model_name', 'k_shots'])['accuracy'].agg(['mean', 'std']) * 100
    print(summary.to_string())


if __name__ == "__main__":
    main()
