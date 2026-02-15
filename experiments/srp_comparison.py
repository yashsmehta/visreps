"""
Temporary script: Compare RSA scores WITH vs WITHOUT Sparse Random Projection.
Model: 32-class AlexNet (seed 1), NSD subject 0, early visual stream.

Reports:
  1. RSA score per layer (with SRP vs without SRP)
  2. Wall-clock time for each pipeline
  3. Activation dimensionality per layer (pre-SRP)
"""

import time
import torch
from omegaconf import OmegaConf

from visreps.utils import rprint
import visreps.models.utils as mutils
from visreps.dataloaders.neural import get_neural_loader
from visreps.analysis.alignment import prepare_data_for_alignment
from visreps.analysis.rsa import compute_rsa_split_half_bootstrap


# ── Config ───────────────────────────────────────────────────────────────
BASE_CFG = {
    "mode": "eval",
    "load_model_from": "checkpoint",
    "seed": 1,
    "cfg_id": 32,
    "checkpoint_dir": "/data/ymehta3/alexnet_pca",
    "checkpoint_model": "checkpoint_epoch_20.pth",
    "pca_labels": True,
    "pca_n_classes": 32,
    "neural_dataset": "nsd",
    "nsd_type": "streams_shared",
    "region": "early visual stream",
    "subject_idx": 0,
    "analysis": "rsa",
    "make_rsm_correlation": "Pearson",
    "compare_rsm_correlation": "Spearman",
    "return_nodes": ["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2"],
    "batchsize": 256,
    "num_workers": 32,
    "reconstruct_from_pcs": False,
    "pca_k": 1,
    "log_expdata": False,
}


def run_pipeline(cfg, label):
    """Run full eval pipeline and return (results, dims, timing)."""
    rprint(f"\n{'='*60}", style="info")
    rprint(f"  Running: {label}  (apply_srp={cfg.apply_srp})", style="info")
    rprint(f"{'='*60}", style="info")

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model + feature extractor (shared across runs, but timed for fairness)
    model = mutils.configure_feature_extractor(cfg, mutils.load_model(cfg, dev))

    # Load neural data
    neural_data, dl = get_neural_loader(cfg)

    # Extract activations (timed)
    t0 = time.perf_counter()
    acts, ids = mutils.get_activations(model, dl, dev, cfg.apply_srp)
    t_extract = time.perf_counter() - t0

    # Align
    acts_aligned, neural_aligned = prepare_data_for_alignment(cfg, acts, neural_data, ids)

    # Collect layer dims
    dims = {}
    for layer, a in acts_aligned.items():
        flat = a.flatten(start_dim=1) if a.ndim > 2 else a
        dims[layer] = flat.shape[1]

    # Split-half bootstrap RSA (returns single-element list with score + CI)
    t1 = time.perf_counter()
    results = compute_rsa_split_half_bootstrap(
        cfg, acts_aligned, neural_aligned, n_bootstrap=1000, seed=42
    )
    t_rsa = time.perf_counter() - t1

    return results, dims, {"extract": t_extract, "rsa": t_rsa, "total": t_extract + t_rsa}


def print_comparison(results_no_srp, results_srp, dims_no_srp, times_no_srp, times_srp):
    """Print a side-by-side comparison of model-level RSA scores, dims, and timing."""
    # Each results list is a single-element list from the bootstrap pipeline
    r_no = results_no_srp[0]
    r_yes = results_srp[0]

    print(f"\n{'='*72}")
    print(f"  SRP Comparison  (32-class AlexNet, NSD subj 0, EVC)")
    print(f"{'='*72}")

    print(f"\n  Activation dimensions per layer:")
    print(f"  {'Layer':<10} {'Dims':>8}")
    print(f"  {'-'*10} {'-'*8}")
    for layer, dim in dims_no_srp.items():
        print(f"  {layer:<10} {dim:>8,}")

    print(f"\n{'='*72}")
    print(f"  Model-level RSA (split-half layer selection + bootstrap)")
    print(f"{'='*72}")
    print(f"  {'':15} {'No SRP':>20}  {'With SRP':>20}")
    print(f"  {'':15} {'-'*20}  {'-'*20}")
    print(f"  {'Best layer':<15} {r_no['layer']:>20}  {r_yes['layer']:>20}")
    print(f"  {'RSA score':<15} {r_no['score']:>20.4f}  {r_yes['score']:>20.4f}")
    print(f"  {'95% CI':<15} [{r_no['ci_low']:.4f}, {r_no['ci_high']:.4f}]"
          f"  [{r_yes['ci_low']:.4f}, {r_yes['ci_high']:.4f}]")

    print(f"\n{'='*72}")
    print(f"  Timing (seconds)")
    print(f"{'='*72}")
    print(f"  {'Phase':<15} {'No SRP':>10}  {'With SRP':>10}  {'Speedup':>10}")
    print(f"  {'-'*15} {'-'*10}  {'-'*10}  {'-'*10}")
    for phase in ["extract", "rsa", "total"]:
        t_no = times_no_srp[phase]
        t_yes = times_srp[phase]
        speedup = t_no / t_yes if t_yes > 0 else float("inf")
        print(f"  {phase:<15} {t_no:>10.2f}  {t_yes:>10.2f}  {speedup:>9.1f}x")
    print()


def main():
    cfg_base = OmegaConf.create(BASE_CFG)

    # ── Run WITHOUT SRP ──────────────────────────────────────────────────
    cfg_no_srp = OmegaConf.merge(cfg_base, {"apply_srp": False})
    results_no_srp, dims_no_srp, times_no_srp = run_pipeline(cfg_no_srp, "WITHOUT SRP")

    # ── Run WITH SRP ─────────────────────────────────────────────────────
    cfg_srp = OmegaConf.merge(cfg_base, {"apply_srp": True})
    results_srp, dims_srp, times_srp = run_pipeline(cfg_srp, "WITH SRP")

    # ── Print comparison ─────────────────────────────────────────────────
    print_comparison(results_no_srp, results_srp, dims_no_srp, times_no_srp, times_srp)


if __name__ == "__main__":
    main()
