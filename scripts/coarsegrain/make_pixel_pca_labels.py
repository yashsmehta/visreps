"""
Generate PCA-based coarse-grained labels from raw pixel statistics.

Negative control: instead of using learned representations (AlexNet, DINO, etc.),
this script computes PCA on flattened raw pixels (resized to 64x64) and applies
global median splits to create 2, 4, 8, ... classes.

Two-pass streaming approach (no large intermediate features file):
  Pass 1: Compute exact mean and covariance incrementally -> eigendecompose
  Pass 2: Project images onto PCs -> median-split -> save labels
"""

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchvision import transforms
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv()

from visreps.dataloaders.obj_cls import get_obj_cls_loader

# Configuration
PIXEL_RES = 64
N_PCS = 6
BATCH_SIZE = 512
NUM_WORKERS = 16
OUTPUT_DIR = "pca_labels/pca_labels_pixels"


def get_loader():
    """Get a single DataLoader over all ImageNet images (no train/test split)."""
    data_cfg = {
        "dataset": "imagenet",
        "batchsize": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "data_augment": False,
        "pca_labels_folder": "N/A",
    }
    preprocess = transforms.Compose([
        transforms.Resize(PIXEL_RES),
        transforms.CenterCrop(PIXEL_RES),
        transforms.ToTensor(),  # [0, 1], no ImageNet normalization
    ])
    _, loaders = get_obj_cls_loader(data_cfg, shuffle=False, train_test_split=False)
    loader = loaders["all"]
    loader.dataset.transform = preprocess
    return loader


def collect_image_names(loader):
    """Extract filenames from the dataset."""
    dataset = loader.dataset
    return [os.path.basename(s[2]) for s in dataset.samples]


def pass1_covariance(loader):
    """Pass 1: Compute exact mean and covariance via streaming batches."""
    p = PIXEL_RES * PIXEL_RES * 3
    running_sum = np.zeros(p, dtype=np.float64)
    n = 0

    print("Pass 1a: Computing mean...")
    for images, _ in tqdm(loader, desc="Mean", unit="batch"):
        batch = images.numpy().reshape(images.shape[0], -1).astype(np.float64)
        running_sum += batch.sum(axis=0)
        n += batch.shape[0]
    mean = running_sum / n

    print(f"Pass 1b: Computing covariance (n={n}, p={p})...")
    cov = np.zeros((p, p), dtype=np.float64)
    for images, _ in tqdm(loader, desc="Covariance", unit="batch"):
        batch = images.numpy().reshape(images.shape[0], -1).astype(np.float64) - mean
        cov += batch.T @ batch
    cov /= n - 1

    print("Computing eigendecomposition...")
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # Sort descending
    idx = np.argsort(eigenvalues)[::-1][:N_PCS]
    eigenvectors = eigenvectors[:, idx]
    eigenvalues = eigenvalues[idx]

    var_explained = eigenvalues.sum() / np.trace(cov) * 100
    print(f"Variance explained by top {N_PCS} PCs: {var_explained:.2f}%")

    return mean, eigenvectors, eigenvalues


def pass2_project_and_label(loader, mean, eigenvectors, image_names):
    """Pass 2: Project images onto PCs, then median-split to create labels."""
    print("Pass 2: Projecting onto PCs...")
    scores_list = []
    for images, _ in tqdm(loader, desc="Project", unit="batch"):
        batch = images.numpy().reshape(images.shape[0], -1).astype(np.float64) - mean
        scores_list.append(batch @ eigenvectors)
    scores = np.concatenate(scores_list, axis=0)

    # Median-split labels (same logic as make_pca_labels.py)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    binary = (scores > np.median(scores, axis=0)).astype(int)

    print("Generating labels...")
    for n_bits in range(1, scores.shape[1] + 1):
        n_classes = 2 ** n_bits
        powers = 2 ** np.arange(n_bits - 1, -1, -1)
        labels = binary[:, :n_bits] @ powers

        df = pd.DataFrame({"image": image_names, "pca_label": labels})
        df.to_csv(os.path.join(OUTPUT_DIR, f"n_classes_{n_classes}.csv"), index=False)

        counts = df["pca_label"].value_counts()
        print(f"  {n_classes:2d} classes: min={counts.min():6d}, max={counts.max():6d}")


def main():
    loader = get_loader()
    image_names = collect_image_names(loader)
    mean, eigenvectors, eigenvalues = pass1_covariance(loader)
    pass2_project_and_label(loader, mean, eigenvectors, image_names)
    print(f"Done. Labels saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
