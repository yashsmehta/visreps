"""
ImageNet-C Robustness: Measures how well representations transfer to corrupted images.

Protocol:
1. Load models from checkpoints
2. Extract features from clean ImageNet validation images
3. Split into train/test; train linear probe on train features
4. Apply all 15 corruptions to test images and evaluate
5. Compare clean vs corrupted accuracy on held-out test set
"""

import os
import sys
import warnings

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
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision.transforms as transforms

from utils import get_device, ensure_output_dir, get_feature_extractor, load_checkpoint_model
from visreps.dataloaders.obj_cls import get_obj_cls_loader

try:
    from imagecorruptions import corrupt
    IMAGECORRUPTIONS_AVAILABLE = True
except ImportError:
    IMAGECORRUPTIONS_AVAILABLE = False
    print("Warning: imagecorruptions not installed. Install with: pip install imagecorruptions")


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
MODELS = {
    "AlexNet (1K classes)": "/data/ymehta3/default/cfg1000a/checkpoint_epoch_20.pth",
    "AlexNet (64 classes)": "/data/ymehta3/alexnet_pca/cfg64a/checkpoint_epoch_20.pth",
    "AlexNet (64→1K curriculum)": "experiments/coarse_grain_benefits/results/curriculum_checkpoints/cfg64_to_1000_late_layers_a/checkpoint_epoch_10.pth",
}

LAYER = "fc2"
N_IMAGES = 5000
SEVERITY = 3
TRAIN_FRACTION = 0.6

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
]


# ─────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────
def load_images_and_labels(loader, n_images):
    """Load raw PIL images and labels from dataset."""
    dataset = loader.dataset
    while hasattr(dataset, 'dataset'):
        dataset = dataset.dataset

    images, labels = [], []
    for i in tqdm(range(min(n_images, len(dataset))), desc="Loading images"):
        try:
            img_path, label, _ = dataset.samples[i]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                img = Image.open(img_path).convert('RGB')
            images.append(img)
            labels.append(label)
        except Exception as e:
            print(f"Warning: Could not load image {i}: {e}")

    return images, np.array(labels)


# ─────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────
PRE_TRANSFORM = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
POST_TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def extract_features_from_images(extractor, images, layer, device, corruption=None, severity=3, batch_size=64):
    """Extract features from PIL images, optionally applying corruption."""
    features_list = []

    for i in tqdm(range(0, len(images), batch_size), desc=f"Extracting {corruption or 'clean'}", leave=False):
        batch_images = images[i:i+batch_size]
        batch_tensors = []

        for img in batch_images:
            img_resized = PRE_TRANSFORM(img)
            img_array = np.array(img_resized)

            if corruption and IMAGECORRUPTIONS_AVAILABLE:
                try:
                    img_array = corrupt(img_array, corruption_name=corruption, severity=severity)
                except Exception:
                    pass

            tensor = POST_TRANSFORM(Image.fromarray(img_array.astype(np.uint8)))
            batch_tensors.append(tensor)

        batch = torch.stack(batch_tensors)

        with torch.no_grad():
            feats = extractor(batch.to(device))[layer]
            if feats.dim() > 2:
                feats = feats.view(feats.size(0), -1)
            features_list.append(feats.cpu().numpy())

    return np.vstack(features_list)


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    if not IMAGECORRUPTIONS_AVAILABLE:
        print("Error: imagecorruptions library required. Install with: pip install imagecorruptions")
        return

    device = get_device()
    print(f"Device: {device}")
    print(f"Corruptions: {CORRUPTIONS}")
    print(f"Severity: {SEVERITY}")

    # Load ImageNet validation data
    print("\n=== Loading ImageNet ===")
    cfg = {"dataset": "imagenet", "batchsize": 256, "num_workers": 4, "pca_labels": False}
    _, loaders = get_obj_cls_loader(cfg, shuffle=False, preprocess=True, train_test_split=True)

    all_images, all_labels = load_images_and_labels(loaders['test'], N_IMAGES)
    print(f"Loaded {len(all_images)} images")

    # Train/test split (probe trained on train, evaluated on test)
    indices = np.arange(len(all_images))
    train_idx, test_idx = train_test_split(indices, train_size=TRAIN_FRACTION, random_state=42, stratify=all_labels)
    train_images = [all_images[i] for i in train_idx]
    test_images = [all_images[i] for i in test_idx]
    train_labels = all_labels[train_idx]
    test_labels = all_labels[test_idx]
    print(f"Train: {len(train_images)}, Test: {len(test_images)}")

    results = []

    for name, checkpoint_path in MODELS.items():
        print(f"\n=== {name} ===")

        try:
            model = load_checkpoint_model(checkpoint_path, device)
        except FileNotFoundError as e:
            print(f"Skipping: {e}")
            continue

        extractor = get_feature_extractor(model, LAYER).to(device).eval()

        # Extract clean features for train and test
        print("  Extracting clean features...")
        train_features = extract_features_from_images(extractor, train_images, LAYER, device)
        test_features = extract_features_from_images(extractor, test_images, LAYER, device)

        # Scale features
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_features)
        test_scaled = scaler.transform(test_features)

        # Train probe on train set, evaluate on test set
        print("  Training linear probe...")
        clf = LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1, verbose=0)
        clf.fit(train_scaled, train_labels)
        clean_acc = clf.score(test_scaled, test_labels)
        print(f"  Clean test accuracy: {clean_acc*100:.2f}%")

        # Evaluate each corruption on test images only
        for corruption in CORRUPTIONS:
            corrupt_features = extract_features_from_images(
                extractor, test_images, LAYER, device, corruption=corruption, severity=SEVERITY
            )
            corrupt_scaled = scaler.transform(corrupt_features)
            corrupt_acc = clf.score(corrupt_scaled, test_labels)
            rel_robust = corrupt_acc / clean_acc if clean_acc > 0 else 0

            print(f"  {corruption}: {corrupt_acc*100:.2f}% (rel: {rel_robust:.3f})")

            results.append({
                'model_name': name,
                'layer': LAYER,
                'corruption': corruption,
                'severity': SEVERITY,
                'clean_acc': clean_acc,
                'corrupt_acc': corrupt_acc,
                'relative_robustness': rel_robust,
            })

        del model, extractor
        torch.cuda.empty_cache()

    # Save
    output_file = os.path.join(ensure_output_dir(), "imagenet_c_robustness.csv")
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")

    # Summary
    print("\n=== Summary ===")
    summary = df.groupby(['model_name', 'corruption']).agg({
        'clean_acc': 'mean',
        'corrupt_acc': 'mean',
        'relative_robustness': 'mean',
    })
    summary[['clean_acc', 'corrupt_acc']] *= 100
    print(summary.to_string())


if __name__ == "__main__":
    main()
