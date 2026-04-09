"""Supplementary Figure S8: Stimulus robustness analysis.

Shows that RSA scores remain stable under stimulus subsampling, confirming
the coarse >= fine-grained alignment result is not driven by a small subset
of stimuli.

For NSD (ventral visual stream) and TVSD (IT), we subsample test stimuli at
fractions from 10% to 100%, repeating 50 times per fraction. Two models are
compared: the best coarse model and the 1000-way baseline.

Usage:
    python manuscript/figures/extended_data/supp_s8_stimulus_robustness.py
    python manuscript/figures/extended_data/supp_s8_stimulus_robustness.py --replot
"""

import argparse
import os
import sys
import sqlite3

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Ensure project root is on the path
ROOT = os.path.dirname(os.path.abspath(__file__))
for _ in range(3):  # go up to project root
    ROOT = os.path.dirname(ROOT)
os.chdir(ROOT)
sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv()

from manuscript.figures.fig_utils import setup_style, GRAN_COLORS, BASELINE_1K_COLOR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FRACTIONS = np.arange(0.1, 1.05, 0.1)  # 0.1, 0.2, ..., 1.0
N_REPS = 50
RNG_SEED = 42
BATCH_SIZE = 128
NUM_WORKERS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Conditions: (label, neural_dataset, region, subject_idx,
#              coarse_cfg_id, coarse_pca_folder, coarse_checkpoint_dir)
CONDITIONS = [
    {
        "label": "NSD: Ventral Visual Stream",
        "neural_dataset": "nsd",
        "region": "ventral visual stream",
        "subject_idx": 0,
        "coarse_cfg_id": 64,
        "coarse_pca_folder": "pca_labels_vit",
        "coarse_checkpoint_dir": "/data/ymehta3/vit_pca",
        "coarse_display": "ViT 64-way",
    },
    {
        "label": "TVSD: IT",
        "neural_dataset": "tvsd",
        "region": "IT",
        "subject_idx": 0,
        "coarse_cfg_id": 64,
        "coarse_pca_folder": "pca_labels_alexnet",
        "coarse_checkpoint_dir": "/data/ymehta3/alexnet_pca",
        "coarse_display": "AlexNet 64-way",
    },
]

BASELINE_CHECKPOINT = "/data/ymehta3/default/cfg1000a/checkpoint_epoch_20.pth"
RETURN_NODES = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2"]

SAVE_DIR = os.path.join(ROOT, "manuscript", "figures", "supplementary", "extra")
DATA_PATH = os.path.join(SAVE_DIR, "supp_s8_data.npz")
FIG_PATH = os.path.join(SAVE_DIR, "stimulus_robustness.png")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_best_layer_from_db(neural_dataset, region, cfg_id, pca_labels_folder, subject_idx=0):
    """Query best layer from layer_selection_scores in results.db."""
    conn = sqlite3.connect("results.db")
    df = pd.read_sql("""
        SELECT l.layer, l.score
        FROM layer_selection_scores l
        JOIN results r ON l.run_id = r.run_id AND l.compare_method = r.compare_method
        WHERE r.neural_dataset = ?
          AND r.region = ?
          AND r.cfg_id = ?
          AND r.pca_labels_folder = ?
          AND r.compare_method = 'spearman'
          AND r.epoch = 20
          AND r.seed = 1
          AND r.subject_idx = ?
          AND r.reconstruct_from_pcs = 0
        ORDER BY l.score DESC
        LIMIT 1
    """, conn, params=[neural_dataset, region, cfg_id, pca_labels_folder, subject_idx])
    conn.close()
    if df.empty:
        raise ValueError(f"No layer selection data for {neural_dataset}/{region}/cfg{cfg_id}/{pca_labels_folder}")
    return df.iloc[0]["layer"]


def load_checkpoint_model(checkpoint_path):
    """Load model from checkpoint, return the raw model (not extractor)."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = checkpoint["model"]
    model.eval()
    return model.to(DEVICE)


def extract_layer_activations(model, dataloader, layer_name):
    """Extract activations for a single layer without SRP."""
    from visreps.models.utils import FeatureExtractor, extract_single_layer

    extractor = FeatureExtractor(
        model, return_nodes=RETURN_NODES, extract_pre_and_post=True
    )
    extractor.eval()
    acts, ids = extract_single_layer(extractor, dataloader, DEVICE, layer_name)
    return acts, ids


def load_neural_test_data(cond):
    """Load test neural data and build a dataloader for the test stimuli."""
    from visreps.dataloaders.neural import (
        load_all_nsd_data, load_all_tvsd_data, _make_loader,
    )
    from visreps.dataloaders.obj_cls import get_transform

    nd = cond["neural_dataset"]
    region = cond["region"]
    subj = cond["subject_idx"]
    transform = get_transform(ds_stats="imgnet")

    if nd == "nsd":
        data = load_all_nsd_data({}, subjects=[subj], regions=[region])
        test_ids = data["shared_test_ids"]
        neural_dict = data["neural"][region][subj]["test"]
        stimuli = data["stimuli"]
    elif nd == "tvsd":
        data = load_all_tvsd_data({}, subjects=[subj], regions=[region])
        test_ids = data["shared_test_ids"]
        neural_dict = data["neural"][region][subj]["test"]
        stimuli = data["stimuli"]
    else:
        raise ValueError(f"Unsupported dataset: {nd}")

    # Build neural matrix aligned to test_ids
    neural_matrix = torch.tensor(
        np.stack([neural_dict[sid] for sid in test_ids]), dtype=torch.float32
    )

    # Build dataloader for test stimuli only
    test_stimuli = {sid: stimuli[sid] for sid in test_ids if sid in stimuli}
    loader = _make_loader(test_stimuli, transform, BATCH_SIZE, NUM_WORKERS)

    return neural_matrix, test_ids, loader


def compute_subsampled_rsa(model_acts, neural_matrix, fractions, n_reps, rng):
    """Compute RSA at various stimulus subsampling fractions.

    Args:
        model_acts: (n_stimuli, n_features) tensor
        neural_matrix: (n_stimuli, n_voxels) tensor
        fractions: array of fractions to test
        n_reps: number of random repetitions per fraction
        rng: numpy RandomState

    Returns:
        results: dict with 'fractions', 'means', 'ci_low', 'ci_high'
    """
    from visreps.analysis.rsa import compute_rdm, compute_rdm_correlation

    n_total = model_acts.shape[0]
    means = []
    ci_lows = []
    ci_highs = []

    # Precompute full RDMs on GPU for speed
    model_acts_gpu = model_acts.to(DEVICE)
    neural_gpu = neural_matrix.to(DEVICE)

    for frac in fractions:
        n_sub = max(int(round(n_total * frac)), 2)
        scores = np.empty(n_reps if frac < 1.0 else 1)

        if frac >= 1.0:
            # Full dataset — single computation
            m_rdm = compute_rdm(model_acts_gpu)
            n_rdm = compute_rdm(neural_gpu)
            scores[0] = compute_rdm_correlation(m_rdm, n_rdm, correlation="Spearman")
        else:
            for rep in range(n_reps):
                idx = torch.from_numpy(
                    rng.choice(n_total, size=n_sub, replace=False)
                ).to(DEVICE)
                m_rdm = compute_rdm(model_acts_gpu[idx])
                n_rdm = compute_rdm(neural_gpu[idx])
                scores[rep] = compute_rdm_correlation(m_rdm, n_rdm, correlation="Spearman")

        means.append(np.mean(scores))
        ci_lows.append(np.percentile(scores, 2.5))
        ci_highs.append(np.percentile(scores, 97.5))
        print(f"    frac={frac:.1f} (n={n_sub}): mean={means[-1]:.4f} "
              f"[{ci_lows[-1]:.4f}, {ci_highs[-1]:.4f}]")

    return {
        "fractions": fractions,
        "means": np.array(means),
        "ci_low": np.array(ci_lows),
        "ci_high": np.array(ci_highs),
    }


# ---------------------------------------------------------------------------
# Main computation
# ---------------------------------------------------------------------------
def run_analysis():
    """Run the full stimulus robustness analysis for all conditions."""
    all_results = {}

    for cond in CONDITIONS:
        label = cond["label"]
        nd = cond["neural_dataset"]
        region = cond["region"]
        subj = cond["subject_idx"]
        print(f"\n{'='*60}")
        print(f"Condition: {label}")
        print(f"{'='*60}")

        # Load neural data and test stimuli
        print("Loading neural data...")
        neural_matrix, test_ids, loader = load_neural_test_data(cond)
        n_test = len(test_ids)
        print(f"  {n_test} test stimuli, {neural_matrix.shape[1]} voxels/channels")

        # --- Coarse model ---
        coarse_cfg = cond["coarse_cfg_id"]
        coarse_folder = cond["coarse_pca_folder"]
        coarse_ckpt = os.path.join(
            cond["coarse_checkpoint_dir"],
            f"cfg{coarse_cfg}a",
            "checkpoint_epoch_20.pth",
        )
        coarse_best_layer = get_best_layer_from_db(
            nd, region, coarse_cfg, coarse_folder, subj
        )
        print(f"\nCoarse model: {cond['coarse_display']} | best layer: {coarse_best_layer}")
        print(f"  Loading from {coarse_ckpt}")
        coarse_model = load_checkpoint_model(coarse_ckpt)
        print("  Extracting activations...")
        coarse_acts, coarse_ids = extract_layer_activations(
            coarse_model, loader, coarse_best_layer
        )
        # Align activations to test_ids order
        id_to_idx = {str(k): i for i, k in enumerate(coarse_ids)}
        coarse_order = [id_to_idx[str(sid)] for sid in test_ids]
        coarse_acts_aligned = coarse_acts[coarse_order]
        del coarse_model
        torch.cuda.empty_cache()

        # --- 1000-way model ---
        baseline_best_layer = get_best_layer_from_db(
            nd, region, 1000, "imagenet1k", subj
        )
        print(f"\n1000-way model | best layer: {baseline_best_layer}")
        print(f"  Loading from {BASELINE_CHECKPOINT}")
        baseline_model = load_checkpoint_model(BASELINE_CHECKPOINT)
        print("  Extracting activations...")
        baseline_acts, baseline_ids = extract_layer_activations(
            baseline_model, loader, baseline_best_layer
        )
        id_to_idx_b = {str(k): i for i, k in enumerate(baseline_ids)}
        baseline_order = [id_to_idx_b[str(sid)] for sid in test_ids]
        baseline_acts_aligned = baseline_acts[baseline_order]
        del baseline_model
        torch.cuda.empty_cache()

        # --- Subsampled RSA ---
        rng = np.random.RandomState(RNG_SEED)
        print(f"\nComputing subsampled RSA (coarse: {cond['coarse_display']})...")
        coarse_results = compute_subsampled_rsa(
            coarse_acts_aligned, neural_matrix, FRACTIONS, N_REPS, rng
        )

        rng = np.random.RandomState(RNG_SEED)
        print(f"\nComputing subsampled RSA (1000-way)...")
        baseline_results = compute_subsampled_rsa(
            baseline_acts_aligned, neural_matrix, FRACTIONS, N_REPS, rng
        )

        key = f"{nd}_{region}".replace(" ", "_")
        all_results[key] = {
            "label": label,
            "coarse_display": cond["coarse_display"],
            "n_test": n_test,
            "fractions": FRACTIONS,
            "coarse_means": coarse_results["means"],
            "coarse_ci_low": coarse_results["ci_low"],
            "coarse_ci_high": coarse_results["ci_high"],
            "baseline_means": baseline_results["means"],
            "baseline_ci_low": baseline_results["ci_low"],
            "baseline_ci_high": baseline_results["ci_high"],
        }

        # Free memory
        del coarse_acts, coarse_acts_aligned, baseline_acts, baseline_acts_aligned
        del neural_matrix
        torch.cuda.empty_cache()

    # Save intermediate data
    save_dict = {}
    for key, res in all_results.items():
        for subkey, val in res.items():
            save_dict[f"{key}__{subkey}"] = val
    np.savez(DATA_PATH, **save_dict)
    print(f"\nSaved intermediate data to {DATA_PATH}")

    return all_results


def load_results():
    """Load previously saved results from .npz file."""
    data = np.load(DATA_PATH, allow_pickle=True)
    all_results = {}
    # Reconstruct nested dict
    for full_key in data.files:
        parts = full_key.split("__", 1)
        cond_key, subkey = parts[0], parts[1]
        if cond_key not in all_results:
            all_results[cond_key] = {}
        val = data[full_key]
        # Convert 0-d arrays back to scalars/strings
        if val.ndim == 0:
            val = val.item()
        all_results[cond_key][subkey] = val
    return all_results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_results(all_results):
    """Create the stimulus robustness figure."""
    setup_style()

    n_panels = len(all_results)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 3.5))
    if n_panels == 1:
        axes = [axes]

    for ax, (key, res) in zip(axes, all_results.items()):
        fracs = res["fractions"]
        n_test = res["n_test"]
        x_vals = (fracs * n_test).astype(int)

        # Coarse model
        coarse_color = GRAN_COLORS[64]  # dark blue
        ax.fill_between(
            x_vals, res["coarse_ci_low"], res["coarse_ci_high"],
            color=coarse_color, alpha=0.15, zorder=2,
        )
        ax.plot(
            x_vals, res["coarse_means"], "-o",
            color=coarse_color, markersize=4, linewidth=1.5,
            markeredgecolor="white", markeredgewidth=0.5,
            label=res["coarse_display"], zorder=3,
        )

        # 1000-way
        ax.fill_between(
            x_vals, res["baseline_ci_low"], res["baseline_ci_high"],
            color=BASELINE_1K_COLOR, alpha=0.15, zorder=2,
        )
        ax.plot(
            x_vals, res["baseline_means"], "-s",
            color=BASELINE_1K_COLOR, markersize=4, linewidth=1.5,
            markeredgecolor="white", markeredgewidth=0.5,
            label="1000-way", zorder=3,
        )

        ax.set_xlabel("Number of test stimuli", fontsize=9, labelpad=4)
        if ax == axes[0]:
            ax.set_ylabel(r"Spearman $\rho$", fontsize=9, labelpad=4)
        ax.set_title(res["label"], fontsize=10, fontweight="semibold", pad=6)

        ax.legend(fontsize=7.5, loc="lower right", frameon=True, framealpha=0.8,
                  edgecolor="#CCCCCC", borderpad=0.4, handlelength=1.5)

        import seaborn as sns
        ax.yaxis.grid(True, which="major", color="#EBEBEB", linewidth=0.4, zorder=0)
        from matplotlib.ticker import AutoMinorLocator
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        ax.tick_params(axis="y", which="major", direction="out", length=4, width=1.0)
        ax.tick_params(axis="y", which="minor", direction="out", length=2.5, width=0.6)
        sns.despine(ax=ax, right=True, top=True, offset=4)

    fig.tight_layout(w_pad=3)
    fig.savefig(FIG_PATH, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"Saved figure to {FIG_PATH}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--replot", action="store_true",
                        help="Skip computation, just re-plot from saved data")
    args = parser.parse_args()

    if args.replot and os.path.exists(DATA_PATH):
        print("Re-plotting from saved data...")
        results = load_results()
    else:
        results = run_analysis()

    plot_results(results)
