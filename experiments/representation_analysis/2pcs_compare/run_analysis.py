"""
Extract features from pretrained and coarse-trained AlexNet across multiple layers,
compute PCA, assign quadrant classes, and save results for plotting.

Layers extracted: conv4, fc1, fc2 (all L2-normalized).
Conv layers are spatially pooled (adaptive avg pool to 3x3) before flattening.

Usage (from project root):
    python experiments/representation_analysis/2pcs_compare/run_analysis.py --n_classes 4
    python experiments/representation_analysis/2pcs_compare/run_analysis.py --n_classes 4 --seed 2
"""
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from dotenv import load_dotenv
load_dotenv()

from visreps.models import custom_model
sys.modules['visreps.models.custom_cnn'] = custom_model

from visreps.dataloaders.obj_cls import get_obj_cls_loader
from visreps.models.utils import FeatureExtractor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LAYERS = ['conv4', 'fc1', 'fc2']


def get_dataloader(dataset='imagenet-mini-50', batch_size=512):
    """Get dataloader for all images in the dataset (no train/test split)."""
    data_cfg = {
        "dataset": dataset,
        "batchsize": batch_size,
        "num_workers": 16,
        "data_augment": False,
        "pca_labels_folder": "N/A",
    }
    _, loaders = get_obj_cls_loader(data_cfg, shuffle=False, train_test_split=False)
    loader = loaders['all']
    loader.dataset.transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return loader


def load_pretrained_alexnet(device):
    """Load pretrained AlexNet (full model, not truncated)."""
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    return model.eval().to(device)


def load_trained_alexnet(checkpoint_dir, n_classes, seed, device):
    """Load coarse-trained CustomCNN from checkpoint (full model)."""
    seed_letter = {1: 'a', 2: 'b', 3: 'c'}[seed]
    path = os.path.join(checkpoint_dir, f'cfg{n_classes}{seed_letter}', 'checkpoint_epoch_20.pth')
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return checkpoint['model'].eval().to(device)


def extract_features(model, loader, device, layers, pool_size=3):
    """Extract L2-normalized features from multiple layers in a single forward pass.

    Conv layers are spatially pooled to pool_size x pool_size before flattening.
    All features are L2-normalized.
    """
    extractor = FeatureExtractor(model, return_nodes=layers,
                                 extract_pre_and_post=False, post_relu=True)
    extractor.to(device).eval()
    pool = nn.AdaptiveAvgPool2d((pool_size, pool_size))

    features = {name: [] for name in layers}
    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Extracting layers", unit="batch"):
            feats = extractor(images.to(device))
            for name in layers:
                out = feats[name]
                if out.dim() == 4:
                    out = pool(out)
                out = out.view(out.size(0), -1)
                out = F.normalize(out, p=2, dim=-1)
                features[name].append(out.cpu())

    return {name: torch.cat(features[name]).numpy() for name in layers}


def compute_pca(features, n_pcs=2):
    """Compute PCA on features. Returns (projections, variance_explained)."""
    mean = features.mean(axis=0)
    centered = features - mean
    cov = (centered.T @ centered) / (len(features) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1][:n_pcs]
    var_explained = eigenvalues[idx] / eigenvalues.sum() * 100
    return centered @ eigenvectors[:, idx], var_explained


def assign_quadrants(pc1, pc2):
    """Assign 4 quadrants via median splits on PC1 and PC2.

    Q0: low PC1, low PC2
    Q1: low PC1, high PC2
    Q2: high PC1, low PC2
    Q3: high PC1, high PC2
    """
    pc1_med, pc2_med = np.median(pc1), np.median(pc2)
    quadrants = np.zeros(len(pc1), dtype=int)
    quadrants[(pc1 <= pc1_med) & (pc2 > pc2_med)] = 1
    quadrants[(pc1 > pc1_med) & (pc2 <= pc2_med)] = 2
    quadrants[(pc1 > pc1_med) & (pc2 > pc2_med)] = 3
    return quadrants, pc1_med, pc2_med


def align_pcs(trained_pcs, trained_var, quadrants):
    """Align trained PCs (order + signs) to match pretrained quadrant layout.

    PCA eigenvectors have two ambiguities: (1) sign (v and -v are both valid),
    and (2) ordering when eigenvalues are close. This tries all 8 configurations
    (2 orderings x 2 sign flips each) and picks the one where quadrant centroids
    best match the expected layout:
        Q0: lower-left, Q1: upper-left, Q2: lower-right, Q3: upper-right.
    """
    expected_signs = np.array([
        [-1, -1], [-1, +1], [+1, -1], [+1, +1],
    ])
    centroids = np.array([trained_pcs[quadrants == q].mean(axis=0) for q in range(4)])

    best_score = -np.inf
    best_config = (False, 1, 1)

    for swap in [False, True]:
        for s1 in [1, -1]:
            for s2 in [1, -1]:
                c = centroids.copy()
                if swap:
                    c = c[:, [1, 0]]
                c[:, 0] *= s1
                c[:, 1] *= s2
                score = (c * expected_signs).sum()
                if score > best_score:
                    best_score = score
                    best_config = (swap, s1, s2)

    swap, s1, s2 = best_config
    if swap:
        trained_pcs = trained_pcs[:, [1, 0]]
        trained_var = trained_var[[1, 0]]
        print("    Swapped PC1 <-> PC2")
    if s1 == -1:
        trained_pcs[:, 0] *= -1
        print("    Flipped PC1 sign")
    if s2 == -1:
        trained_pcs[:, 1] *= -1
        print("    Flipped PC2 sign")

    return trained_pcs, trained_var


def main():
    parser = argparse.ArgumentParser(
        description="Extract features, compute PCA across layers, save results for plotting"
    )
    parser.add_argument('--n_classes', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1, choices=[1, 2, 3])
    parser.add_argument('--checkpoint_dir', type=str,
                        default='/data/ymehta3/alexnet_pca/')
    parser.add_argument('--dataset', type=str, default='imagenet-mini-50')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = get_dataloader(args.dataset)

    # --- Pretrained AlexNet (1000-way) ---
    print("Extracting pretrained AlexNet...")
    pretrained_model = load_pretrained_alexnet(device)
    pretrained_feats = extract_features(pretrained_model, loader, device, LAYERS)
    del pretrained_model
    torch.cuda.empty_cache()

    # --- Coarse-trained AlexNet ---
    print(f"\nExtracting {args.n_classes}-way trained AlexNet (seed {args.seed})...")
    trained_model = load_trained_alexnet(
        args.checkpoint_dir, args.n_classes, args.seed, device
    )
    trained_feats = extract_features(trained_model, loader, device, LAYERS)
    del trained_model
    torch.cuda.empty_cache()

    # --- PCA per layer ---
    save_dict = {'n_classes': args.n_classes, 'layers': np.array(LAYERS)}

    for layer in LAYERS:
        print(f"\n--- {layer} ---")
        print(f"  Pretrained: {pretrained_feats[layer].shape}, "
              f"Trained: {trained_feats[layer].shape}")

        p_pcs, p_var = compute_pca(pretrained_feats[layer])
        t_pcs, t_var = compute_pca(trained_feats[layer])
        print(f"  Pretrained var: PC1={p_var[0]:.1f}%, PC2={p_var[1]:.1f}%")
        print(f"  Trained var:    PC1={t_var[0]:.1f}%, PC2={t_var[1]:.1f}%")

        # Quadrants from pretrained PCA at this layer
        quadrants, pc1_med, pc2_med = assign_quadrants(p_pcs[:, 0], p_pcs[:, 1])

        # Align trained PCs to match pretrained quadrant layout
        print("  Aligning trained PCs:")
        t_pcs, t_var = align_pcs(t_pcs, t_var, quadrants)

        save_dict[f'{layer}_pretrained_pcs'] = p_pcs
        save_dict[f'{layer}_trained_pcs'] = t_pcs
        save_dict[f'{layer}_pretrained_var'] = p_var
        save_dict[f'{layer}_trained_var'] = t_var
        save_dict[f'{layer}_quadrants'] = quadrants
        save_dict[f'{layer}_pretrained_medians'] = np.array([pc1_med, pc2_med])

    output_path = os.path.join(SCRIPT_DIR, f'data_{args.n_classes}way.npz')
    np.savez_compressed(output_path, **save_dict)
    print(f"\nSaved analysis data to {output_path}")


if __name__ == '__main__':
    main()
