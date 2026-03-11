"""
Extract features from pretrained and coarse-trained AlexNet,
project each onto its OWN top 2 PCs, and color by the actual PCA labels.

Each model is projected onto its own principal component axes, revealing its intrinsic
geometry. Images are colored by PCA labels (the coarse model's training signal).

Features are L2-normalized by default to match the PCA label generation pipeline
(see scripts/extract_representations/alexnet_representations.py).

Usage (from project root):
    python experiments/representation_analysis/2pcs_compare/run_analysis.py
    python experiments/representation_analysis/2pcs_compare/run_analysis.py --n_classes 2 --pca_labels_folder pca_labels_clip --checkpoint_dir /data/ymehta3/clip_pca/
    python experiments/representation_analysis/2pcs_compare/run_analysis.py --n_classes 4 --seed 2
    python experiments/representation_analysis/2pcs_compare/run_analysis.py --no_l2_norm  # raw activations
"""
import os
import sys
import argparse
import numpy as np
import pandas as pd
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
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../../..'))
LAYERS = ['conv4', 'fc1', 'fc2']


def get_dataloader(dataset='imagenet-mini-50', batch_size=512):
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


def load_pca_labels(loader, n_classes, pca_labels_folder):
    """Load PCA labels from CSV and match to dataset image order."""
    csv_path = os.path.join(PROJECT_ROOT,
                            f'pca_labels/{pca_labels_folder}/n_classes_{n_classes}.csv')
    df = pd.read_csv(csv_path)
    label_map = dict(zip(df['image'], df['pca_label']))

    labels = []
    for sample in loader.dataset.samples:
        img_id = os.path.basename(sample[2])
        labels.append(label_map.get(img_id, -1))

    labels = np.array(labels)
    n_valid = (labels >= 0).sum()
    print(f"PCA labels ({pca_labels_folder}, {n_classes}-way): "
          f"{n_valid}/{len(labels)} matched ({n_valid/len(labels)*100:.1f}%)")
    return labels


def load_pretrained_alexnet(device):
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    return model.eval().to(device)


def load_trained_alexnet(checkpoint_dir, n_classes, seed, device):
    seed_letter = {1: 'a', 2: 'b', 3: 'c'}[seed]
    path = os.path.join(checkpoint_dir, f'cfg{n_classes}{seed_letter}',
                        'checkpoint_epoch_20.pth')
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return checkpoint['model'].eval().to(device)


def extract_features(model, loader, device, layers, pool_size=3, l2_norm=False):
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
                out = out.flatten(1)
                if l2_norm:
                    out = F.normalize(out, p=2, dim=-1)
                features[name].append(out.cpu())

    return {name: torch.cat(features[name]).numpy() for name in layers}


def compute_pca(features, n_pcs=2):
    mean = features.mean(axis=0)
    centered = features - mean
    cov = (centered.T @ centered) / (len(features) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1][:n_pcs]
    var_explained = eigenvalues[idx] / eigenvalues.sum() * 100
    components = eigenvectors[:, idx]
    return centered @ components, components, mean, var_explained


def main():
    parser = argparse.ArgumentParser(
        description="Extract features, compute per-model PCA, save results"
    )
    parser.add_argument('--n_classes', type=int, default=4)
    parser.add_argument('--seed', type=int, default=1, choices=[1, 2, 3])
    parser.add_argument('--checkpoint_dir', type=str,
                        default='/data/ymehta3/alexnet_pca/')
    parser.add_argument('--pca_labels_folder', type=str,
                        default='pca_labels_alexnet')
    parser.add_argument('--dataset', type=str, default='imagenet-mini-50')
    parser.add_argument('--batch_size', type=int, default=512)
    parser.add_argument('--layers', type=str, nargs='+', default=None,
                        help='Layers to extract (default: conv4 fc1 fc2)')
    parser.add_argument('--l2_norm', action='store_true', default=True,
                        help='L2-normalize features (matches PCA label generation)')
    parser.add_argument('--no_l2_norm', action='store_false', dest='l2_norm',
                        help='Disable L2 normalization')
    args = parser.parse_args()

    layers = args.layers or LAYERS
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    loader = get_dataloader(args.dataset, batch_size=args.batch_size)

    # --- Load PCA labels ---
    print("Loading PCA labels...")
    pca_labels = load_pca_labels(loader, args.n_classes, args.pca_labels_folder)
    valid = pca_labels >= 0

    # --- Save image paths ---
    img_paths = np.array([sample[0] for sample in loader.dataset.samples])

    # --- Pretrained AlexNet ---
    norm_str = " (L2-normed)" if args.l2_norm else ""
    print(f"\nExtracting pretrained AlexNet{norm_str}...")
    pretrained_model = load_pretrained_alexnet(device)
    pretrained_feats = extract_features(pretrained_model, loader, device, layers,
                                        l2_norm=args.l2_norm)
    del pretrained_model
    torch.cuda.empty_cache()

    # --- Coarse-trained AlexNet ---
    print(f"\nExtracting {args.n_classes}-way trained AlexNet (seed {args.seed}){norm_str}...")
    trained_model = load_trained_alexnet(
        args.checkpoint_dir, args.n_classes, args.seed, device
    )
    trained_feats = extract_features(trained_model, loader, device, layers,
                                     l2_norm=args.l2_norm)
    del trained_model
    torch.cuda.empty_cache()

    # --- PCA per layer (each model gets its own PCs) ---
    save_dict = {
        'n_classes': args.n_classes,
        'pca_labels': pca_labels[valid],
        'pca_labels_folder': args.pca_labels_folder,
        'img_paths': img_paths[valid],
    }

    used_layers = []
    for layer in layers:
        p_feats = pretrained_feats[layer][valid]
        t_feats = trained_feats[layer][valid]
        print(f"\n--- {layer} ---")
        print(f"  Pretrained: {p_feats.shape}, Trained: {t_feats.shape}")

        p_pcs, _, _, p_var = compute_pca(p_feats)
        print(f"  Pretrained var: PC1={p_var[0]:.1f}%, PC2={p_var[1]:.1f}%")

        t_pcs, _, _, t_var = compute_pca(t_feats)
        print(f"  Trained var: PC1={t_var[0]:.1f}%, PC2={t_var[1]:.1f}%")

        save_dict[f'{layer}_pretrained_pcs'] = p_pcs
        save_dict[f'{layer}_pretrained_var'] = p_var
        save_dict[f'{layer}_trained_pcs'] = t_pcs
        save_dict[f'{layer}_trained_var'] = t_var
        used_layers.append(layer)

    save_dict['layers'] = np.array(used_layers)

    tag = args.pca_labels_folder.replace('pca_labels_', '')
    output_path = os.path.join(SCRIPT_DIR, f'data_{args.n_classes}way_{tag}.npz')
    np.savez_compressed(output_path, **save_dict)
    print(f"\nSaved: {output_path}")


if __name__ == '__main__':
    main()
