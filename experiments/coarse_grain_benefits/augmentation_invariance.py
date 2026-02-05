"""
Augmentation Invariance: Measures how stable representations are under OOD augmentations.
Uses albumentations for augmentations NOT seen during training (MotionBlur, Shadow, Elastic, etc.)
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
import torchvision.transforms as T
from PIL import Image
import albumentations as A

from utils import get_device, ensure_output_dir, get_feature_extractor, load_checkpoint_model
from visreps.dataloaders.obj_cls import get_obj_cls_loader


# ─────────────────────────────────────────────────────────────
# CONFIG: (name, checkpoint_path) for each model to evaluate
# ─────────────────────────────────────────────────────────────
MODELS = {
    "AlexNet (1K classes)": "/data/ymehta3/default/cfg1000a/checkpoint_epoch_20.pth",
    "AlexNet (64 classes)": "/data/ymehta3/alexnet_pca/cfg64a/checkpoint_epoch_20.pth",
    "AlexNet (64→1K curriculum)": "experiments/coarse_grain_benefits/results/curriculum_checkpoints/cfg64_to_1000_late_layers_a/checkpoint_epoch_10.pth",
}

LAYER = "fc2"
N_IMAGES = 1000
N_AUGMENTS = 10

# OOD augmentations (NOT used during training)
OOD_TRANSFORM = A.Compose([
    A.MotionBlur(blur_limit=15, p=0.5),
    A.RandomShadow(p=0.5),
    A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
    A.GridDistortion(p=0.3),
    A.OpticalDistortion(p=0.3),
    A.GaussNoise(var_limit=(10, 50), p=0.5),
])

# Standard ImageNet normalization (applied after albumentations)
NORMALIZE = T.Compose([
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_pil_images(loader, n_images):
    """Load raw PIL images from dataset."""
    dataset = loader.dataset
    while hasattr(dataset, 'dataset'):
        dataset = dataset.dataset

    images = []
    for i in tqdm(range(min(n_images, len(dataset))), desc="Loading images"):
        try:
            img = Image.open(dataset.samples[i][0]).convert('RGB')
            images.append(img)
        except Exception as e:
            print(f"Warning: Could not load image {i}: {e}")
    return images


def augment_image(pil_img):
    """Apply OOD augmentations and return normalized tensor."""
    # Resize to 224x224 first
    pil_img = pil_img.resize((224, 224))
    img_np = np.array(pil_img)

    # Apply albumentations
    augmented = OOD_TRANSFORM(image=img_np)["image"]

    # Convert to tensor and normalize
    return NORMALIZE(Image.fromarray(augmented))


def cosine_similarity_matrix(features):
    """Compute pairwise cosine similarity."""
    normalized = features / np.linalg.norm(features, axis=1, keepdims=True).clip(min=1e-8)
    return normalized @ normalized.T


def compute_invariance(model, images, layer, n_augments, device):
    """Compute mean augmentation invariance across images."""
    extractor = get_feature_extractor(model, layer).to(device).eval()
    scores = []

    for img in tqdm(images, desc="Computing invariance", leave=False):
        # Generate augmented batch
        batch = torch.stack([augment_image(img) for _ in range(n_augments)])

        # Extract features
        with torch.no_grad():
            feats = extractor(batch.to(device))[layer]
            feats = feats.view(feats.size(0), -1).cpu().numpy()

        # Mean pairwise similarity (upper triangle, excluding diagonal)
        sim = cosine_similarity_matrix(feats)
        triu_idx = np.triu_indices(n_augments, k=1)
        scores.append(sim[triu_idx].mean())

    return np.mean(scores), np.std(scores)


def main():
    device = get_device()
    print(f"Device: {device}")
    print("OOD augmentations: MotionBlur, RandomShadow, ElasticTransform, GridDistortion, OpticalDistortion, GaussNoise")

    # Load images
    print("\n=== Loading ImageNet ===")
    cfg = {"dataset": "imagenet", "batchsize": 256, "num_workers": 4, "pca_labels": False}
    _, loaders = get_obj_cls_loader(cfg, shuffle=False, preprocess=True, train_test_split=False)
    images = load_pil_images(loaders['all'], N_IMAGES)
    print(f"Loaded {len(images)} images")

    # Evaluate models
    results = []

    for name, checkpoint_path in MODELS.items():
        print(f"\n=== {name} ===")

        try:
            model = load_checkpoint_model(checkpoint_path, device)
        except FileNotFoundError as e:
            print(f"Skipping: {e}")
            continue

        mean_inv, std_inv = compute_invariance(model, images, LAYER, N_AUGMENTS, device)
        print(f"  {LAYER}: mean={mean_inv:.4f}, std={std_inv:.4f}")

        results.append({
            'model_name': name,
            'layer': LAYER,
            'mean_invariance': mean_inv,
            'std_invariance': std_inv,
        })

        del model
        torch.cuda.empty_cache()

    # Save
    output_file = os.path.join(ensure_output_dir(), "augmentation_invariance.csv")
    pd.DataFrame(results).to_csv(output_file, index=False)
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    main()
