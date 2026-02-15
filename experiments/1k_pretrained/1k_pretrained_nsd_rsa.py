"""
Standalone experiment: Pretrained model RSA on NSD

Run from repo root:
    python experiments/1k_pretrained_nsd_rsa.py

Supports:
- ViT-B/16 and AlexNet pretrained on ImageNet-1K (torchvision)
- Custom AlexNet trained on coarse-grained labels (checkpoint)

Saves results to CSV with normalized depth for cross-architecture comparison.
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from torchvision.models import vit_b_16, ViT_B_16_Weights, alexnet, AlexNet_Weights
from torchvision import transforms
from tqdm import tqdm
from omegaconf import OmegaConf

from visreps.dataloaders.neural import load_nsd_data, _make_loader
from visreps.analysis.alignment import prepare_data_for_alignment
from visreps.analysis.rsa import compute_rsa_alignment
from visreps.analysis.sparse_random_projection import get_srp_transformer
from visreps.utils import rprint, get_seed_letter

# Backward compat for loading checkpoints
from visreps.models import custom_model
sys.modules['visreps.models.custom_cnn'] = custom_model


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
MODEL = "alexnet"  # "vit", "alexnet"
MODEL_SOURCE = "checkpoint"  # "torchvision" or "checkpoint"

# Checkpoint config (only used if MODEL_SOURCE == "checkpoint")
CHECKPOINT_DIR = "/data/ymehta3/default"
CFG_ID = 1000  # e.g., 2, 4, 8, 16, 32, 64, 1000
SEED = 1  # 1, 2, or 3 (maps to a, b, c)
CHECKPOINT_MODEL = "checkpoint_epoch_20.pth"

# NSD config
SUBJECT_IDS = [0, 1, 2, 3, 4, 5, 6, 7]  # NSD subjects (0-7)
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
OUTPUT_CSV = "logs/1k_pretrained_nsd_rsa.csv"


# ─────────────────────────────────────────────────────────────
# MODEL LOADERS
# ─────────────────────────────────────────────────────────────
def load_vit_torchvision():
    """Load ViT-B/16 pretrained on ImageNet-1K."""
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)
    preprocess = weights.transforms()
    return model, preprocess, "torchvision"


def load_alexnet_torchvision():
    """Load AlexNet pretrained on ImageNet-1K."""
    weights = AlexNet_Weights.IMAGENET1K_V1
    model = alexnet(weights=weights)
    preprocess = weights.transforms()
    return model, preprocess, "torchvision"


def load_alexnet_checkpoint():
    """Load custom AlexNet from checkpoint."""
    seed_letter = get_seed_letter(SEED)
    checkpoint_path = f"{CHECKPOINT_DIR}/cfg{CFG_ID}{seed_letter}/{CHECKPOINT_MODEL}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = checkpoint['model']
    print(f"Loaded checkpoint: {checkpoint_path}")

    # Standard ImageNet preprocessing for custom models
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return model, preprocess, CHECKPOINT_DIR


# ─────────────────────────────────────────────────────────────
# HOOK REGISTRATION
# ─────────────────────────────────────────────────────────────
def register_vit_hooks(model):
    """Register hooks on all 12 ViT encoder layers."""
    layer_outputs = {}
    hooks = []
    layer_names = [f"block_{i}" for i in range(12)]

    def make_hook(name):
        def hook(module, input, output):
            layer_outputs[name] = output[:, 0, :].detach()  # CLS token
        return hook

    for i in range(12):
        layer = model.encoder.layers[i]
        h = layer.register_forward_hook(make_hook(f"block_{i}"))
        hooks.append(h)

    return hooks, layer_outputs, layer_names


def register_alexnet_torchvision_hooks(model):
    """Register hooks on torchvision AlexNet layers."""
    layer_outputs = {}
    hooks = []

    # Torchvision AlexNet: 5 conv + 2 fc
    layer_info = [
        ("conv1", model.features[0]),
        ("conv2", model.features[3]),
        ("conv3", model.features[6]),
        ("conv4", model.features[8]),
        ("conv5", model.features[10]),
        ("fc1", model.classifier[1]),
        ("fc2", model.classifier[4]),
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


def register_alexnet_custom_hooks(model):
    """Register hooks on custom AlexNet layers."""
    layer_outputs = {}
    hooks = []

    # CustomCNN: conv layers at 0,4,8,11,14; fc at 1,5
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
    print(f"Model: {MODEL}, Source: {MODEL_SOURCE}")
    print(f"RSA config: make_rsm={MAKE_RSM_CORRELATION}, compare_rsm={COMPARE_RSM_CORRELATION}")

    # Load model based on config
    if MODEL == "vit":
        if MODEL_SOURCE != "torchvision":
            raise ValueError("ViT only supports torchvision source")
        print("Loading ViT-B/16 (ImageNet-1K pretrained)...")
        model, preprocess, model_source = load_vit_torchvision()
        hooks, layer_outputs, layer_names = register_vit_hooks(model)

    elif MODEL == "alexnet":
        if MODEL_SOURCE == "torchvision":
            print("Loading AlexNet (ImageNet-1K pretrained)...")
            model, preprocess, model_source = load_alexnet_torchvision()
            hooks, layer_outputs, layer_names = register_alexnet_torchvision_hooks(model)
        elif MODEL_SOURCE == "checkpoint":
            print("Loading custom AlexNet from checkpoint...")
            model, preprocess, model_source = load_alexnet_checkpoint()
            hooks, layer_outputs, layer_names = register_alexnet_custom_hooks(model)
        else:
            raise ValueError(f"Unknown MODEL_SOURCE: {MODEL_SOURCE}")
    else:
        raise ValueError(f"Unknown model: {MODEL}")

    depth_map = compute_normalized_depth(layer_names)
    print(f"Layers: {layer_names}")
    print(f"Normalized depths: {[f'{depth_map[n]:.3f}' for n in layer_names]}")

    model = model.to(device)
    model.eval()

    all_results = []

    for region in REGIONS:
        for subject_idx in SUBJECT_IDS:
            rprint(f"\n{'='*50}", style="info")
            rprint(f"Subject {subject_idx} | Region: {region}", style="info")
            rprint(f"{'='*50}", style="info")

            # Load NSD fMRI data and stimuli
            cfg_nsd = {"region": region, "subject_idx": subject_idx, "nsd_type": NSD_TYPE}
            targets, stimuli = load_nsd_data(cfg_nsd)
            print(f"Loaded {len(stimuli)} NSD stimuli")

            # Create dataloader
            dataloader = _make_loader(stimuli, preprocess, BATCH_SIZE, NUM_WORKERS)

            # Extract features from all layers
            all_features = {name: [] for name in layer_names}
            all_keys = []
            srp_projectors = {}  # SRP projection matrices per layer

            with torch.no_grad():
                for batch_idx, (images, keys) in enumerate(tqdm(dataloader, desc=f"Extracting {MODEL} features")):
                    images = images.to(device)
                    _ = model(images)  # Forward pass triggers hooks

                    # Initialize SRP on first batch (only for layers with D > SRP_DIM)
                    if APPLY_SRP and batch_idx == 0:
                        for name in layer_names:
                            feat = layer_outputs[name]
                            D = feat.view(feat.size(0), -1).size(1)
                            if D > SRP_DIM:
                                transformer = get_srp_transformer(D, SRP_DIM, None, None, SRP_CACHE_DIR)
                                if transformer is not None:
                                    # Convert to torch sparse tensor on GPU
                                    sparse_matrix = transformer.components_
                                    coo = sparse_matrix.tocoo()
                                    indices = torch.from_numpy(np.vstack([coo.row, coo.col])).long()
                                    values = torch.from_numpy(coo.data).float()
                                    proj_matrix = torch.sparse_coo_tensor(indices, values, coo.shape).to(device)
                                    srp_projectors[name] = proj_matrix
                        if srp_projectors:
                            rprint(f"Initialized SRP projectors for {len(srp_projectors)} layers (D > {SRP_DIM})", style="success")

                    for name in layer_names:
                        feat = layer_outputs[name]
                        feat = feat.view(feat.size(0), -1).float()

                        # Apply SRP if enabled
                        if APPLY_SRP and name in srp_projectors:
                            feat = torch.sparse.mm(srp_projectors[name], feat.t()).t()

                        feat = feat / feat.norm(dim=-1, keepdim=True)  # L2 norm
                        all_features[name].append(feat.cpu())

                    all_keys.extend(keys)

            # Concatenate features
            acts = {name: torch.cat(all_features[name], dim=0) for name in layer_names}
            print(f"Features shape (first layer): {acts[layer_names[0]].shape}")

            # Align activations with neural data
            cfg_align = OmegaConf.create({"neural_dataset": "nsd"})
            acts_aligned, neural_aligned = prepare_data_for_alignment(
                cfg_align, acts, targets, all_keys
            )
            print(f"Aligned samples: {neural_aligned.shape[0]}")

            # Compute RSA for each layer
            cfg_rsa = OmegaConf.create({
                "make_rsm_correlation": MAKE_RSM_CORRELATION,
                "compare_rsm_correlation": COMPARE_RSM_CORRELATION,
            })
            results = compute_rsa_alignment(cfg_rsa, acts_aligned, neural_aligned)

            # Collect results
            rprint(f"\n{'─'*40}", style="highlight")
            rprint(f"{MODEL.upper()} RSA Scores by Layer:", style="highlight")
            rprint(f"{'─'*40}", style="highlight")

            for r in results:
                layer = r["layer"]
                score = r["score"]
                depth = depth_map[layer]
                print(f"  {layer:10s} (depth={depth:.3f}): {score:.4f}")

                all_results.append({
                    "model": MODEL,
                    "layer": layer,
                    "depth_normalized": depth,
                    "rsa_score": score,
                    "subject_id": subject_idx,
                    "region": region,
                    "model_source": model_source,
                })

    # Save to CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df = pd.DataFrame(all_results)

    # Append if file exists, otherwise create new
    if os.path.exists(OUTPUT_CSV):
        df_existing = pd.read_csv(OUTPUT_CSV)
        df = pd.concat([df_existing, df], ignore_index=True)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV}")

    # Clean up hooks
    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()
