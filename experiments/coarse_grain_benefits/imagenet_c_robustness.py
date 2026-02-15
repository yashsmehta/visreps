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
from concurrent.futures import ThreadPoolExecutor
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
BATCH_SIZE = 512
NUM_WORKERS = 8

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
# IMAGE PREPROCESSING
# ─────────────────────────────────────────────────────────────
PRE_TRANSFORM = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224)])
NORMALIZE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def preprocess_to_arrays(images):
    """Resize and crop all PIL images to numpy arrays (done once)."""
    return [np.array(PRE_TRANSFORM(img)) for img in tqdm(images, desc="Pre-resizing")]


def prepare_tensors(image_arrays, corruption=None, severity=3):
    """Corrupt and normalize images in parallel using threads."""
    def process(arr):
        if corruption:
            try:
                arr = corrupt(arr, corruption_name=corruption, severity=severity)
            except Exception:
                pass
        return NORMALIZE(Image.fromarray(arr.astype(np.uint8)))

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        tensors = list(tqdm(
            pool.map(process, image_arrays),
            total=len(image_arrays),
            desc=f"Preparing {corruption or 'clean'}",
            leave=False,
        ))

    return torch.stack(tensors)


# ─────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────
def extract_features(extractor, tensors, layer, device, batch_size=BATCH_SIZE):
    """Extract features from pre-processed tensors in large GPU batches."""
    features_list = []
    for i in range(0, len(tensors), batch_size):
        batch = tensors[i:i + batch_size]
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
    print(f"Batch size: {BATCH_SIZE}, Workers: {NUM_WORKERS}")
    print(f"Corruptions: {len(CORRUPTIONS)}, Severity: {SEVERITY}")

    # Load ImageNet validation data
    print("\n=== Loading ImageNet ===")
    cfg = {"dataset": "imagenet", "batchsize": 256, "num_workers": 4, "pca_labels": False}
    _, loaders = get_obj_cls_loader(cfg, shuffle=False, preprocess=True, train_test_split=True)

    all_images, all_labels = load_images_and_labels(loaders['test'], N_IMAGES)
    print(f"Loaded {len(all_images)} images")

    # Pre-resize all images to numpy arrays (done once, reused for every corruption)
    all_arrays = preprocess_to_arrays(all_images)
    del all_images

    # Train/test split
    indices = np.arange(len(all_arrays))
    train_idx, test_idx = train_test_split(indices, train_size=TRAIN_FRACTION, random_state=42)
    train_arrays = [all_arrays[i] for i in train_idx]
    test_arrays = [all_arrays[i] for i in test_idx]
    train_labels = all_labels[train_idx]
    test_labels = all_labels[test_idx]
    print(f"Train: {len(train_arrays)}, Test: {len(test_arrays)}")

    # Pre-compute clean tensors once (reused across all models)
    print("\n=== Preparing clean tensors ===")
    train_tensors = prepare_tensors(train_arrays)
    test_tensors = prepare_tensors(test_arrays)

    # Phase 1: Train probes for all models on clean features
    print("\n=== Phase 1: Training probes ===")
    trained_models = {}
    results = []

    for name, checkpoint_path in MODELS.items():
        print(f"\n--- {name} ---")
        try:
            model = load_checkpoint_model(checkpoint_path, device)
        except FileNotFoundError as e:
            print(f"Skipping: {e}")
            continue

        extractor = get_feature_extractor(model, LAYER).to(device).eval()

        train_features = extract_features(extractor, train_tensors, LAYER, device)
        test_features = extract_features(extractor, test_tensors, LAYER, device)

        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(train_features)
        test_scaled = scaler.transform(test_features)

        print("  Training linear probe...")
        clf = LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1, verbose=0)
        clf.fit(train_scaled, train_labels)
        clean_acc = clf.score(test_scaled, test_labels)
        print(f"  Clean test accuracy: {clean_acc * 100:.2f}%")

        trained_models[name] = {
            'extractor': extractor,
            'scaler': scaler,
            'clf': clf,
            'clean_acc': clean_acc,
        }
        torch.cuda.empty_cache()

    del train_tensors, test_tensors

    # Phase 2: Evaluate corruptions (each prepared once, evaluated on all models)
    print("\n=== Phase 2: Evaluating corruptions ===")
    for corruption in CORRUPTIONS:
        print(f"\n--- {corruption} ---")
        corrupt_tensors = prepare_tensors(test_arrays, corruption=corruption, severity=SEVERITY)

        for name, info in trained_models.items():
            corrupt_features = extract_features(info['extractor'], corrupt_tensors, LAYER, device)
            corrupt_scaled = info['scaler'].transform(corrupt_features)
            corrupt_acc = info['clf'].score(corrupt_scaled, test_labels)
            rel_robust = corrupt_acc / info['clean_acc'] if info['clean_acc'] > 0 else 0

            print(f"  {name}: {corrupt_acc * 100:.2f}% (rel: {rel_robust:.3f})")

            results.append({
                'model_name': name,
                'layer': LAYER,
                'corruption': corruption,
                'severity': SEVERITY,
                'clean_acc': info['clean_acc'],
                'corrupt_acc': corrupt_acc,
                'relative_robustness': rel_robust,
            })

        del corrupt_tensors
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
