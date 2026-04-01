"""Compute and cache PCA projections for Figure 1 label space scatter.

Projects 1 image per ImageNet class (1000 points) onto saved CLIP PCA axes.

Usage (from project root):
    python manuscript/figures/fig1/compute_pca_cache.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()

from visreps.dataloaders.obj_cls import get_obj_cls_loader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(SCRIPT_DIR, "pc_scatter_1per_class.npz")

EIGVEC_PATH = "datasets/obj_cls/imagenet/eigenvectors_clip.npz"
DATASET = "imagenet-mini-50"


def _get_dataloader(dataset=DATASET, batch_size=128):
    """Load ImageNet with CLIP preprocessing."""
    import clip
    _, preprocess = clip.load("ViT-L/14", device="cpu")
    data_cfg = {
        "dataset": dataset,
        "batchsize": batch_size,
        "num_workers": 16,
        "data_augment": False,
        "pca_labels_folder": "N/A",
    }
    _, loaders = get_obj_cls_loader(data_cfg, shuffle=False, train_test_split=False)
    loader = loaders["all"]
    loader.dataset.transform = preprocess
    return loader


def _extract_clip_features(loader, device):
    """Extract CLIP ViT-L/14 image features, L2-normalized."""
    import clip
    model, _ = clip.load("ViT-L/14", device=device)
    model.eval()
    features = []
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Extracting CLIP", unit="batch"):
            out = model.encode_image(images.to(device))
            out = out / out.norm(dim=-1, keepdim=True)
            features.append(out.float().cpu())
    del model
    torch.cuda.empty_cache()
    return torch.cat(features).numpy()


def compute_and_cache():
    """Extract CLIP features, project onto PCA axes, cache 1 image per class."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading eigenvectors from {EIGVEC_PATH}...")
    eigdata = np.load(EIGVEC_PATH)
    eigenvectors = eigdata["eigenvectors"][:, :2]
    mean = eigdata["mean"]
    eigenvalues = eigdata["eigenvalues"][:2]
    total_var = float(eigdata["total_variance"])
    var_explained = eigenvalues / total_var * 100
    print(f"  PC1: {var_explained[0]:.2f}%, PC2: {var_explained[1]:.2f}%")

    loader = _get_dataloader()
    imagenet_labels = np.array([s[1] for s in loader.dataset.samples])

    print("\nExtracting CLIP ViT-L/14 features...")
    features = _extract_clip_features(loader, device)
    pcs_all = (features - mean) @ eigenvectors

    print("\nSelecting 1 image per class (1000 total)...")
    selected = []
    for c in range(1000):
        class_mask = np.where(imagenet_labels == c)[0]
        if len(class_mask) == 0:
            continue
        class_pcs = pcs_all[class_mask]
        centroid = class_pcs.mean(axis=0)
        dists = np.linalg.norm(class_pcs - centroid, axis=1)
        selected.append(class_mask[np.argmin(dists)])

    selected = np.array(selected)
    print(f"  Selected {len(selected)} images")

    np.savez_compressed(CACHE_PATH, pcs=pcs_all[selected],
                        var_explained=var_explained,
                        class_labels=imagenet_labels[selected])
    print(f"\nCached -> {CACHE_PATH}")


if __name__ == "__main__":
    compute_and_cache()
