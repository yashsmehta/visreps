"""
Curriculum RSA Experiment: Compare 3 AlexNet variants on NSD

Compares RSA alignment for:
1. AlexNet trained from scratch on 1K-way ImageNet
2. AlexNet trained on 64-way coarse labels
3. AlexNet pre-trained on 64-way, then fine-tuned on 1K-way (curriculum, late_layers)

Run from repo root:
    python experiments/coarse_grain_benefits/curriculum_nsd_rsa.py
"""

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from tqdm import tqdm
from omegaconf import OmegaConf

from visreps.dataloaders.neural import load_nsd_data, _make_loader
from visreps.analysis.alignment import prepare_data_for_alignment
from visreps.analysis.rsa import compute_rsa_alignment
from visreps.analysis.sparse_random_projection import get_srp_transformer
from visreps.utils import rprint

# Backward compat for loading checkpoints
from visreps.models import custom_model
sys.modules['visreps.models.custom_cnn'] = custom_model


# ─────────────────────────────────────────────────────────────
# MODEL CONFIGS
# ─────────────────────────────────────────────────────────────
MODELS = {
    "AlexNet (1K classes)": "/data/ymehta3/default/cfg1000a/checkpoint_epoch_20.pth",
    "AlexNet (64 classes)": "/data/ymehta3/alexnet_pca/cfg64a/checkpoint_epoch_20.pth",
    "AlexNet (64→1K curriculum)": os.path.join(
        PROJECT_ROOT,
        "experiments/coarse_grain_benefits/results/curriculum_checkpoints",
        "cfg64_to_1000_late_layers_a/checkpoint_epoch_10.pth",
    ),
}

# NSD config
SUBJECT_IDS = [0, 1, 2, 3, 4, 5, 6, 7]
REGIONS = ["early visual stream", "ventral visual stream"]
NSD_TYPE = "streams_shared"

BATCH_SIZE = 64
NUM_WORKERS = 4

# RSA settings
MAKE_RSM_CORRELATION = "Pearson"
COMPARE_RSM_CORRELATION = "Spearman"

# SRP settings (applied only when layer dim > SRP_DIM)
APPLY_SRP = True
SRP_DIM = 4096
SRP_CACHE_DIR = "model_checkpoints/srp_cache"

# Output
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "results")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "curriculum_nsd_rsa.csv")
OUTPUT_PNG = os.path.join(OUTPUT_DIR, "curriculum_rsa_comparison.png")

# Plot style
COLORS = {
    "AlexNet (1K classes)": "#0072B2",
    "AlexNet (64 classes)": "#2E8B57",
    "AlexNet (64→1K curriculum)": "#D55E00",
}
LINESTYLES = {
    "AlexNet (1K classes)": "-",
    "AlexNet (64 classes)": "--",
    "AlexNet (64→1K curriculum)": "-.",
}
MARKERS = {
    "AlexNet (1K classes)": "o",
    "AlexNet (64 classes)": "o",
    "AlexNet (64→1K curriculum)": "D",
}


# ─────────────────────────────────────────────────────────────
# HOOKS (Custom AlexNet architecture)
# ─────────────────────────────────────────────────────────────
def register_alexnet_custom_hooks(model):
    """Register hooks on custom AlexNet layers."""
    layer_outputs = {}
    hooks = []

    layer_info = [
        ("conv1", model.features[0]),
        ("conv2", model.features[4]),
        ("conv3", model.features[8]),
        ("conv4", model.features[11]),
        ("conv5", model.features[14]),
        ("fc1", model.classifier[1]),
        ("fc2", model.classifier[5]),
    ]
    layer_names = [name for name, _ in layer_info]

    def make_hook(name):
        def hook(module, input, output):
            if output.dim() == 4:
                output = output.flatten(start_dim=1)
            layer_outputs[name] = output.detach()
        return hook

    for name, layer in layer_info:
        h = layer.register_forward_hook(make_hook(name))
        hooks.append(h)

    return hooks, layer_outputs, layer_names


def compute_normalized_depth(layer_names):
    """Compute normalized depth (0.0 to 1.0) for each layer."""
    n = len(layer_names)
    if n == 1:
        return {layer_names[0]: 1.0}
    return {name: i / (n - 1) for i, name in enumerate(layer_names)}


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"RSA config: make_rsm={MAKE_RSM_CORRELATION}, compare_rsm={COMPARE_RSM_CORRELATION}")

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    all_results = []

    for model_name, checkpoint_path in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"{'='*60}")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = checkpoint['model']

        hooks, layer_outputs, layer_names = register_alexnet_custom_hooks(model)
        depth_map = compute_normalized_depth(layer_names)
        print(f"Layers: {layer_names}")

        model = model.to(device)
        model.eval()

        for region in REGIONS:
            for subject_idx in SUBJECT_IDS:
                rprint(f"\n  Subject {subject_idx} | Region: {region}", style="info")

                cfg_nsd = {"region": region, "subject_idx": subject_idx, "nsd_type": NSD_TYPE}
                targets, stimuli = load_nsd_data(cfg_nsd)

                dataloader = _make_loader(stimuli, preprocess, BATCH_SIZE, NUM_WORKERS)

                all_features = {name: [] for name in layer_names}
                all_keys = []
                srp_projectors = {}

                with torch.no_grad():
                    for batch_idx, (images, keys) in enumerate(tqdm(dataloader, desc="Extracting features")):
                        images = images.to(device)
                        _ = model(images)

                        # Initialize SRP on first batch
                        if APPLY_SRP and batch_idx == 0:
                            for name in layer_names:
                                feat = layer_outputs[name]
                                D = feat.view(feat.size(0), -1).size(1)
                                if D > SRP_DIM:
                                    transformer = get_srp_transformer(D, SRP_DIM, None, None, SRP_CACHE_DIR)
                                    if transformer is not None:
                                        sparse_matrix = transformer.components_
                                        coo = sparse_matrix.tocoo()
                                        indices = torch.from_numpy(np.vstack([coo.row, coo.col])).long()
                                        values = torch.from_numpy(coo.data).float()
                                        proj_matrix = torch.sparse_coo_tensor(indices, values, coo.shape).to(device)
                                        srp_projectors[name] = proj_matrix

                        for name in layer_names:
                            feat = layer_outputs[name]
                            feat = feat.view(feat.size(0), -1).float()

                            if APPLY_SRP and name in srp_projectors:
                                feat = torch.sparse.mm(srp_projectors[name], feat.t()).t()

                            feat = feat / feat.norm(dim=-1, keepdim=True)
                            all_features[name].append(feat.cpu())

                        all_keys.extend(keys)

                acts = {name: torch.cat(all_features[name], dim=0) for name in layer_names}

                cfg_align = OmegaConf.create({"neural_dataset": "nsd"})
                acts_aligned, neural_aligned = prepare_data_for_alignment(
                    cfg_align, acts, targets, all_keys
                )

                cfg_rsa = OmegaConf.create({
                    "make_rsm_correlation": MAKE_RSM_CORRELATION,
                    "compare_rsm_correlation": COMPARE_RSM_CORRELATION,
                })
                results = compute_rsa_alignment(cfg_rsa, acts_aligned, neural_aligned)

                for r in results:
                    layer = r["layer"]
                    score = r["score"]
                    depth = depth_map[layer]
                    print(f"    {layer:10s} (depth={depth:.3f}): {score:.4f}")

                    all_results.append({
                        "model_name": model_name,
                        "layer": layer,
                        "depth_normalized": depth,
                        "rsa_score": score,
                        "subject_id": subject_idx,
                        "region": region,
                    })

        # Clean up hooks before loading next model
        for h in hooks:
            h.remove()

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV}")

    # Plot
    plot_results(df)


# ─────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────
def plot_results(df):
    """Two-panel RSA-by-depth plot (EVC, VVS) for all 3 models."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 7,
        'axes.titlesize': 8,
        'axes.labelsize': 7,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'legend.fontsize': 6,
        'axes.linewidth': 0.5,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'figure.dpi': 300,
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75))
    plt.subplots_adjust(wspace=0.2, left=0.07, right=0.72, top=0.88, bottom=0.18)

    all_lines, all_labels = [], []

    for ax, region, ylabel in [
        (axes[0], "early visual stream", True),
        (axes[1], "ventral visual stream", False),
    ]:
        for model_name in MODELS:
            mask = (df["model_name"] == model_name) & (df["region"] == region)
            grouped = df[mask].groupby("depth_normalized")["rsa_score"].mean()
            depths, means = grouped.index.values, grouped.values
            sort_idx = np.argsort(depths)

            line, = ax.plot(
                depths[sort_idx], means[sort_idx],
                color=COLORS[model_name],
                marker=MARKERS[model_name],
                linestyle=LINESTYLES[model_name],
                markersize=4, markerfacecolor=COLORS[model_name],
                markeredgecolor='white', markeredgewidth=0.4,
                linewidth=1.5, zorder=3,
            )
            if region == "early visual stream":
                all_lines.append(line)
                all_labels.append(model_name)

        ax.set_xlabel("Normalized depth")
        if ylabel:
            ax.set_ylabel("RSA score")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(0, None)

        region_short = "Early Visual" if "early" in region else "Ventral Visual"
        ax.set_title(region_short, fontweight='bold', pad=4)

    fig.legend(
        all_lines, all_labels,
        loc='center right', bbox_to_anchor=(0.98, 0.5),
        frameon=True, framealpha=0.95, edgecolor='none',
    )

    for i, ax in enumerate(axes):
        ax.text(-0.15, 1.08, chr(97 + i), transform=ax.transAxes,
                fontsize=10, fontweight='bold', va='top')

    fig.savefig(OUTPUT_PNG, format='png', dpi=300, bbox_inches='tight')
    print(f"Plot saved to {OUTPUT_PNG}")
    plt.close(fig)


if __name__ == "__main__":
    main()
