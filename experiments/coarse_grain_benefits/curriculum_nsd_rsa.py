"""
Curriculum RSA Experiment: Compare 3 AlexNet variants on NSD

Compares RSA alignment for:
1. AlexNet trained from scratch on 1K-way ImageNet
2. AlexNet trained on 64-way coarse labels
3. AlexNet pre-trained on 64-way, then fine-tuned on 1K-way (curriculum, late_layers)

Uses the same two-phase RSA pipeline as the main eval system:
  Phase 1: Layer selection on train stimuli using SRP activations
  Phase 2: Re-extract best layer without SRP for exact test RDMs

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
from omegaconf import OmegaConf

from visreps.dataloaders.neural import load_all_nsd_data, _make_loader
from visreps.dataloaders.obj_cls import get_transform
from visreps.analysis.alignment import _align_stimulus_level
from visreps.analysis.rsa import compute_rdm, compute_rdm_correlation
import visreps.models.utils as mutils
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
RETURN_NODES = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2"]
BATCH_SIZE = 256
NUM_WORKERS = 8

# RSA settings (matches standard pipeline)
COMPARE_METHOD = "Spearman"  # passed directly to compute_rdm_correlation
N_SELECT = 1000              # subsample train stimuli for layer selection

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


# ─────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"RSA: compare_method={COMPARE_METHOD}, n_select={N_SELECT}")

    # Config for FeatureExtractor (extract_pre_and_post=True gives _pre/_post layers)
    extractor_cfg = OmegaConf.create({
        "return_nodes": RETURN_NODES,
        "extract_pre_and_post": True,
    })

    # Load all NSD data once (shared across models)
    rprint("Loading NSD data for all subjects and regions...", style="info")
    nsd_cfg = OmegaConf.create({"neural_dataset": "nsd"})
    all_data = load_all_nsd_data(nsd_cfg, subjects=SUBJECT_IDS, regions=REGIONS)
    neural = all_data["neural"]
    stimuli = all_data["stimuli"]
    shared_test_ids = all_data["shared_test_ids"]
    rprint(
        f"  {len(SUBJECT_IDS)} subjects x {len(REGIONS)} regions, "
        f"{len(stimuli)} stimuli, {len(shared_test_ids)} shared test IDs",
        style="success",
    )

    # Build dataloaders (shared across models)
    transform = get_transform(ds_stats="imgnet")
    dl_all = _make_loader(stimuli, transform, BATCH_SIZE, NUM_WORKERS)
    test_stimuli = {sid: stimuli[sid] for sid in shared_test_ids if sid in stimuli}
    dl_test = _make_loader(test_stimuli, transform, BATCH_SIZE, NUM_WORKERS)

    all_results = []

    for model_name, checkpoint_path in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"{'='*60}")

        # Load model from checkpoint, wrap with FeatureExtractor
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = mutils.configure_feature_extractor(extractor_cfg, checkpoint['model'])
        model = model.to(device)
        model.eval()

        # ══════════════════════════════════════════════════════
        # Extract activations with SRP (standard pipeline)
        # ══════════════════════════════════════════════════════
        rprint("  Extracting activations with SRP...", style="info")
        acts, ids = mutils.get_activations(model, dl_all, device)
        rprint(f"  Extracted {len(acts)} layers, {len(ids)} stimuli", style="success")

        # ══════════════════════════════════════════════════════
        # PHASE 1: Per-subject layer selection (uses SRP activations)
        # ══════════════════════════════════════════════════════
        rprint("\n  Phase 1: Per-subject layer selection", style="info")
        per_region_layers = {}

        for region in REGIONS:
            per_region_layers[region] = {}

            for subj in SUBJECT_IDS:
                subj_neural_train = neural[region][subj]["train"]
                train_acts, train_neural, _ = _align_stimulus_level(
                    acts, subj_neural_train, ids
                )

                n_train = train_neural.size(0)
                if N_SELECT < n_train:
                    rng_sel = np.random.RandomState(42)
                    sel_idx = rng_sel.choice(n_train, size=N_SELECT, replace=False)
                else:
                    sel_idx = np.arange(n_train)

                neural_rdm_sel = compute_rdm(train_neural[sel_idx])

                best_layer, best_score = None, -float("inf")
                for layer, layer_acts in train_acts.items():
                    flat = layer_acts[sel_idx].flatten(start_dim=1) if layer_acts.ndim > 2 else layer_acts[sel_idx]
                    layer_rdm = compute_rdm(flat)
                    score = compute_rdm_correlation(
                        layer_rdm, neural_rdm_sel, correlation=COMPARE_METHOD
                    )
                    if score > best_score:
                        best_score = score
                        best_layer = layer

                per_region_layers[region][subj] = best_layer
                rprint(
                    f"    {region} subj {subj}: {best_layer} ({best_score:.4f})",
                    style="info",
                )

                del train_acts, train_neural

        # Free bulk SRP activations
        del acts
        torch.cuda.empty_cache()

        # ══════════════════════════════════════════════════════
        # PHASE 2: Re-extract best layers without SRP, score on test
        # ══════════════════════════════════════════════════════
        rprint("\n  Phase 2: Test evaluation (re-extract without SRP)", style="info")

        # Collect unique best layers, re-extract each without SRP
        all_unique_layers = set()
        for region_layers in per_region_layers.values():
            all_unique_layers.update(region_layers.values())

        model_rdms = {}
        for layer in sorted(all_unique_layers):
            exact_acts, _ = mutils.extract_single_layer(
                model, dl_test, device, layer, shared_test_ids
            )
            flat = exact_acts.flatten(start_dim=1) if exact_acts.ndim > 2 else exact_acts
            model_rdms[layer] = compute_rdm(flat)
            del exact_acts

        # Free model
        del model
        torch.cuda.empty_cache()

        # Per-(region, subject) scoring
        for region in REGIONS:
            for subj in SUBJECT_IDS:
                best_layer = per_region_layers[region][subj]

                # Build neural test RDM
                test_neural_dict = neural[region][subj]["test"]
                responses = [
                    test_neural_dict[sid]
                    for sid in shared_test_ids
                    if sid in test_neural_dict
                ]
                neural_tensor = torch.as_tensor(
                    np.stack(responses).squeeze(), dtype=torch.float32
                )
                neural_rdm = compute_rdm(neural_tensor)

                score = compute_rdm_correlation(
                    model_rdms[best_layer], neural_rdm,
                    correlation=COMPARE_METHOD,
                )

                rprint(
                    f"    {region} subj {subj} | {best_layer} = {score:.4f}",
                    style="highlight",
                )

                all_results.append({
                    "model_name": model_name,
                    "best_layer": best_layer,
                    "test_score": score,
                    "subject_id": subj,
                    "region": region,
                })

        # Cleanup between models
        del model_rdms, per_region_layers

    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.DataFrame(all_results)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nResults saved to {OUTPUT_CSV}")

    plot_results(df)


# ─────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────
def plot_results(df):
    """Two-panel bar chart (EVC, VVS) showing mean ± std test RSA across subjects."""
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

    model_names = list(MODELS.keys())
    all_bars, all_labels = [], []

    for ax, region, ylabel in [
        (axes[0], "early visual stream", True),
        (axes[1], "ventral visual stream", False),
    ]:
        for i, model_name in enumerate(model_names):
            sub = df[(df["model_name"] == model_name) & (df["region"] == region)]
            if sub.empty:
                continue

            mean_score = sub["test_score"].mean()
            std_score = sub["test_score"].std()

            bar = ax.bar(
                i, mean_score, yerr=std_score,
                color=COLORS[model_name],
                capsize=3, width=0.6, alpha=0.85,
                edgecolor='white', linewidth=0.5,
            )
            if region == "early visual stream":
                all_bars.append(bar[0])
                all_labels.append(model_name)

        ax.set_xticks([])
        if ylabel:
            ax.set_ylabel("RSA score (test)")
        ax.set_ylim(0, None)

        region_short = "Early Visual" if "early" in region else "Ventral Visual"
        ax.set_title(region_short, fontweight='bold', pad=4)

    fig.legend(
        all_bars, all_labels,
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
