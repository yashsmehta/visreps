"""
Inspect what the top pixel PCs capture by visualizing extreme images.

Samples ~50K random ImageNet images, computes PCA on flattened 64x64 pixels,
then saves a grid of the top/bottom 10 images for PC1, PC2, PC3.
"""

import os
import sys
import numpy as np
from scipy.sparse.linalg import eigsh
from PIL import Image
import torch
from torchvision import transforms
from dotenv import load_dotenv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
load_dotenv()

from visreps.dataloaders.obj_cls import get_obj_cls_loader

PIXEL_RES = 64
N_PCS = 3
N_EXTREME = 10
N_SAMPLE = 20000
OUTPUT_DIR = "scripts/coarsegrain"
DISPLAY_RES = 128  # Resolution for the output grid images


def main():
    # Load dataset
    data_cfg = {
        "dataset": "imagenet",
        "batchsize": 512,
        "num_workers": 16,
        "data_augment": False,
        "pca_labels_folder": "N/A",
    }
    preprocess = transforms.Compose([
        transforms.Resize(PIXEL_RES),
        transforms.CenterCrop(PIXEL_RES),
        transforms.ToTensor(),
    ])
    _, loaders = get_obj_cls_loader(data_cfg, shuffle=False, train_test_split=False)
    loader = loaders["all"]
    loader.dataset.transform = preprocess
    dataset = loader.dataset

    n_total = len(dataset)
    print(f"Total images: {n_total}, sampling {N_SAMPLE}", flush=True)

    # Random subset indices
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(n_total, size=N_SAMPLE, replace=False)
    sample_idx.sort()

    # Collect pixel features for sample
    print("Loading sample images...")
    p = PIXEL_RES * PIXEL_RES * 3
    features = np.zeros((N_SAMPLE, p), dtype=np.float32)
    for i, idx in enumerate(sample_idx):
        img_tensor, _ = dataset[idx]
        features[i] = img_tensor.numpy().reshape(-1)
        if (i + 1) % 10000 == 0:
            print(f"  {i+1}/{N_SAMPLE}")

    # PCA via partial eigendecomposition (only top N_PCS, much faster than full eigh)
    print("Computing PCA (partial eigsh)...")
    mean = features.mean(axis=0)
    centered = features - mean
    cov = (centered.astype(np.float64).T @ centered.astype(np.float64)) / (N_SAMPLE - 1)
    print("  Covariance computed, running eigsh...")
    eigenvalues, eigenvectors = eigsh(cov, k=N_PCS, which='LM')
    # eigsh returns ascending order, flip to descending
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]

    total_var = np.trace(cov)
    for i in range(N_PCS):
        print(f"  PC{i+1}: {eigenvalues[i]/total_var*100:.2f}% variance explained")

    # Project
    scores = centered @ eigenvectors.astype(np.float32)

    # For each PC, find top/bottom images and build grid
    high_res_transform = transforms.Compose([
        transforms.Resize(DISPLAY_RES),
        transforms.CenterCrop(DISPLAY_RES),
    ])

    for pc in range(N_PCS):
        pc_scores = scores[:, pc]
        bottom_idx = np.argsort(pc_scores)[:N_EXTREME]
        top_idx = np.argsort(pc_scores)[-N_EXTREME:][::-1]

        # Print score ranges
        print(f"\nPC{pc+1}:")
        print(f"  Top 10 scores:    {pc_scores[top_idx[0]]:.3f} to {pc_scores[top_idx[-1]]:.3f}")
        print(f"  Bottom 10 scores: {pc_scores[bottom_idx[0]]:.3f} to {pc_scores[bottom_idx[-1]]:.3f}")

        # Build image grid: 2 rows (top, bottom) x 10 cols
        grid = Image.new("RGB", (DISPLAY_RES * N_EXTREME, DISPLAY_RES * 2))

        for col, si in enumerate(top_idx):
            orig_idx = sample_idx[si]
            img_path = dataset.samples[orig_idx][0]
            img = Image.open(img_path).convert("RGB")
            img = high_res_transform(img)
            grid.paste(img, (col * DISPLAY_RES, 0))

        for col, si in enumerate(bottom_idx):
            orig_idx = sample_idx[si]
            img_path = dataset.samples[orig_idx][0]
            img = Image.open(img_path).convert("RGB")
            img = high_res_transform(img)
            grid.paste(img, (col * DISPLAY_RES, DISPLAY_RES))

        out_path = os.path.join(OUTPUT_DIR, f"pixel_pc{pc+1}_extremes.png")
        grid.save(out_path)
        print(f"  Saved: {out_path}")

        # Also print the image filenames for reference
        print(f"  Top 10 images (high PC{pc+1}):")
        for si in top_idx:
            orig_idx = sample_idx[si]
            fname = dataset.samples[orig_idx][2]
            print(f"    {fname} (score={pc_scores[si]:.3f})")
        print(f"  Bottom 10 images (low PC{pc+1}):")
        for si in bottom_idx:
            orig_idx = sample_idx[si]
            fname = dataset.samples[orig_idx][2]
            print(f"    {fname} (score={pc_scores[si]:.3f})")

    # Also compute and print mean pixel values for extremes to quantify
    print("\n--- Quantitative summary ---")
    for pc in range(N_PCS):
        pc_scores = scores[:, pc]
        top_idx = np.argsort(pc_scores)[-N_EXTREME:]
        bottom_idx = np.argsort(pc_scores)[:N_EXTREME]

        top_feats = features[top_idx].reshape(N_EXTREME, 3, PIXEL_RES, PIXEL_RES)
        bot_feats = features[bottom_idx].reshape(N_EXTREME, 3, PIXEL_RES, PIXEL_RES)

        # Mean per-channel intensity
        top_rgb = top_feats.mean(axis=(0, 2, 3))  # (3,)
        bot_rgb = bot_feats.mean(axis=(0, 2, 3))

        # Overall luminance (approximate)
        top_lum = 0.2126 * top_rgb[0] + 0.7152 * top_rgb[1] + 0.0722 * top_rgb[2]
        bot_lum = 0.2126 * bot_rgb[0] + 0.7152 * bot_rgb[1] + 0.0722 * bot_rgb[2]

        # Spatial frequency (mean gradient magnitude)
        top_grad = np.mean([np.abs(np.diff(f, axis=-1)).mean() + np.abs(np.diff(f, axis=-2)).mean() for f in top_feats])
        bot_grad = np.mean([np.abs(np.diff(f, axis=-1)).mean() + np.abs(np.diff(f, axis=-2)).mean() for f in bot_feats])

        print(f"\nPC{pc+1}:")
        print(f"  Top 10:    R={top_rgb[0]:.3f} G={top_rgb[1]:.3f} B={top_rgb[2]:.3f}  Lum={top_lum:.3f}  Grad={top_grad:.4f}")
        print(f"  Bottom 10: R={bot_rgb[0]:.3f} G={bot_rgb[1]:.3f} B={bot_rgb[2]:.3f}  Lum={bot_lum:.3f}  Grad={bot_grad:.4f}")


if __name__ == "__main__":
    main()
