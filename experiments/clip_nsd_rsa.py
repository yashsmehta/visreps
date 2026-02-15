"""
Standalone experiment: ViT RSA on NSD (ventral visual stream)

Run from repo root:
    python experiments/clip_nsd_rsa.py

Uses torchvision ViT-B/16 pretrained on ImageNet-1K.
Reports RSA scores for all 12 transformer layers.
"""

import torch
from torchvision.models import vit_b_16, ViT_B_16_Weights
from tqdm import tqdm
from omegaconf import OmegaConf

from visreps.dataloaders.neural import load_nsd_data, _make_loader
from visreps.analysis.alignment import prepare_data_for_alignment
from visreps.analysis.rsa import compute_rsa_alignment
from visreps.utils import rprint


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────
SUBJECT_IDS = [0]  # NSD subjects (0-7)
REGION = "ventral visual stream"
NSD_TYPE = "streams_shared"

BATCH_SIZE = 32
NUM_WORKERS = 4

# RSA settings
MAKE_RSM_CORRELATION = "Pearson"
COMPARE_RSM_CORRELATION = "Spearman"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"RSA config: make_rsm={MAKE_RSM_CORRELATION}, compare_rsm={COMPARE_RSM_CORRELATION}")

    # Load ViT-B/16 pretrained on ImageNet-1K
    print("Loading ViT-B/16 (ImageNet-1K pretrained)...")
    weights = ViT_B_16_Weights.IMAGENET1K_V1
    model = vit_b_16(weights=weights)
    preprocess = weights.transforms()

    model = model.to(device)
    model.eval()

    # Register hooks on all 12 encoder layers
    layer_outputs = {}

    def make_hook(name):
        def hook(module, input, output):
            # output shape: [batch, num_tokens, hidden_dim]
            # Extract CLS token (first token) as the layer representation
            layer_outputs[name] = output[:, 0, :].detach()
        return hook

    hooks = []
    for i in range(12):
        layer = model.encoder.layers[i]
        h = layer.register_forward_hook(make_hook(f"block_{i}"))
        hooks.append(h)

    for subject_idx in SUBJECT_IDS:
        rprint(f"\n{'='*50}", style="info")
        rprint(f"Subject {subject_idx} | Region: {REGION}", style="info")
        rprint(f"{'='*50}", style="info")

        # Load NSD fMRI data and stimuli
        cfg_nsd = {"region": REGION, "subject_idx": subject_idx, "nsd_type": NSD_TYPE}
        targets, stimuli = load_nsd_data(cfg_nsd)
        print(f"Loaded {len(stimuli)} NSD stimuli")

        # Create dataloader
        dataloader = _make_loader(stimuli, preprocess, BATCH_SIZE, NUM_WORKERS)

        # Extract features from all layers
        all_features = {f"block_{i}": [] for i in range(12)}
        all_keys = []

        with torch.no_grad():
            for images, keys in tqdm(dataloader, desc="Extracting ViT features"):
                images = images.to(device)
                _ = model(images)  # Forward pass triggers hooks

                for layer_name in all_features:
                    feat = layer_outputs[layer_name]
                    feat = feat / feat.norm(dim=-1, keepdim=True)  # L2 norm
                    all_features[layer_name].append(feat.cpu())

                all_keys.extend(keys)

        # Concatenate features
        acts = {}
        for layer_name in all_features:
            acts[layer_name] = torch.cat(all_features[layer_name], dim=0)
        print(f"Features shape per layer: {acts['block_0'].shape}")

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

        # Print results
        rprint(f"\n{'─'*40}", style="highlight")
        rprint("ViT RSA Scores by Layer:", style="highlight")
        rprint(f"{'─'*40}", style="highlight")
        for r in results:
            print(f"  {r['layer']:10s}: {r['score']:.4f}")

    # Clean up hooks
    for h in hooks:
        h.remove()


if __name__ == "__main__":
    main()
