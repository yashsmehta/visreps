"""
Optimized Reconstruction Analysis
===================================
Sweeps pca_k from 1-15 for the 1000-way model, measuring how much of the
brain-alignment signal is captured by the top-k PCs of layer activations.

Skips Phase 1 (layer selection) by querying results.db for pre-computed best
layers from existing 1000-way evaluations. Extracts each unique best layer
only ONCE per seed, then sweeps all pca_k values using cached activations.

Usage:
    python experiments/reconstruction_analysis/run_reconstruction.py
    python experiments/reconstruction_analysis/run_reconstruction.py --datasets nsd tvsd
"""

import argparse
import sqlite3

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from omegaconf import OmegaConf

from visreps.utils import rprint, save_results
from visreps.evals import _load_cfg
from visreps.models.utils import (
    load_model,
    configure_feature_extractor,
    extract_single_layer,
    get_activations,
)
from visreps.dataloaders.neural import (
    load_all_nsd_data,
    load_all_tvsd_data,
    get_neural_loader,
    _make_loader,
)
from visreps.dataloaders.obj_cls import get_transform
from visreps.analysis.rsa import (
    compute_rdm,
    compute_rdm_correlation,
    _concept_average_exact,
)
from visreps.analysis.alignment import prepare_concept_alignment, AlignmentData
from visreps.analysis.reconstruct_from_pcs import reconstruct_from_pcs


# ── Constants ─────────────────────────────────────────────────────────────────

DB_PATH = "results.db"
CHECKPOINT_DIR = "/data/ymehta3/default"
CFG_ID = 1000
SEEDS = [1, 2, 3]
PCA_K_RANGE = range(1, 16)
COMPARE_METHOD = "spearman"

DATASET_CONFIG = {
    "nsd": {
        "regions": ["early visual stream", "ventral visual stream"],
        "subjects": list(range(8)),
    },
    "tvsd": {
        "regions": ["V1", "V4", "IT"],
        "subjects": [0, 1],
    },
    "things-behavior": {
        "regions": ["N/A"],
        "subjects": ["N/A"],
    },
}


# ── DB query ──────────────────────────────────────────────────────────────────

def query_best_layers(neural_dataset, seed):
    """Query results.db for best layers from existing 1000-way baseline evaluations.

    Returns dict mapping (region, subject_idx_str) -> best_layer_name.
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT region, subject_idx, layer, score
        FROM results
        WHERE cfg_id = ?
          AND checkpoint_dir = ?
          AND reconstruct_from_pcs = 0
          AND analysis = 'rsa'
          AND compare_method = ?
          AND neural_dataset = ?
          AND seed = ?
    """, conn, params=[CFG_ID, CHECKPOINT_DIR, COMPARE_METHOD, neural_dataset, seed])
    conn.close()

    if df.empty:
        raise ValueError(
            f"No baseline results for {neural_dataset} seed={seed}. "
            "Run the standard 1000-way evaluation first."
        )

    # Best layer per (region, subject_idx) — highest score
    idx = df.groupby(["region", "subject_idx"])["score"].idxmax()
    best = df.loc[idx]
    return {(row.region, row.subject_idx): row.layer for _, row in best.iterrows()}


# ── Config ────────────────────────────────────────────────────────────────────

def build_cfg(seed, neural_dataset):
    """Construct eval config and merge with training config from checkpoint."""
    cfg = OmegaConf.create({
        "mode": "eval",
        "neural_dataset": neural_dataset,
        "return_nodes": ["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2"],
        "extract_pre_and_post": True,
        "reconstruct_from_pcs": True,
        "pca_k": 1,
        "load_model_from": "checkpoint",
        "seed": seed,
        "cfg_id": CFG_ID,
        "checkpoint_dir": CHECKPOINT_DIR,
        "checkpoint_model": "checkpoint_epoch_20.pth",
        "analysis": "rsa",
        "compare_method": COMPARE_METHOD,
        "bootstrap": True,
        "n_bootstrap": 1000,
        "batchsize": 256,
        "num_workers": 32,
        "log_expdata": True,
        "verbose": False,
    })
    return _load_cfg(cfg)


# ── Bootstrap helper ──────────────────────────────────────────────────────────

def bootstrap_rdm_correlation(model_rdm, neural_rdm, method, n_bootstrap=1000, seed=42):
    """Point estimate + 1000-iteration bootstrap 95% CI for RDM correlation."""
    score = compute_rdm_correlation(
        model_rdm, neural_rdm, correlation=method.capitalize()
    )

    rng = np.random.RandomState(seed)
    n = neural_rdm.size(0)
    n_sub = int(n * 0.9)
    boot_scores = np.empty(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        idx = torch.from_numpy(rng.choice(n, size=n_sub, replace=False))
        boot_scores[i] = compute_rdm_correlation(
            model_rdm[idx][:, idx],
            neural_rdm[idx][:, idx],
            correlation=method.capitalize(),
        )

    ci_low = float(np.percentile(boot_scores, 2.5))
    ci_high = float(np.percentile(boot_scores, 97.5))
    return score, ci_low, ci_high, boot_scores.tolist()


# ── Result saving helper ─────────────────────────────────────────────────────

def _save(cfg, layer, score, ci_low, ci_high, boot_scores):
    """Build result DataFrame and save to results.db."""
    result_df = pd.DataFrame([{
        "layer": layer,
        "compare_method": COMPARE_METHOD,
        "score": score,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "analysis": "rsa",
        "layer_selection_scores": [],
        "bootstrap_scores": boot_scores,
    }])
    save_results(result_df, cfg)


# ── NSD / TVSD ────────────────────────────────────────────────────────────────

def run_nsd_tvsd(neural_dataset):
    """Run pca_k reconstruction sweep for NSD or TVSD."""
    ds = DATASET_CONFIG[neural_dataset]
    regions, subjects = ds["regions"], ds["subjects"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for seed in SEEDS:
        rprint(f"\n{'='*60}", style="info")
        rprint(f"  {neural_dataset.upper()} | seed {seed}", style="info")
        rprint(f"{'='*60}\n", style="info")

        # 1. Query DB for best layers
        best_layers = query_best_layers(neural_dataset, seed)
        rprint(f"  Best layers from DB:", style="success")
        for region in regions:
            for subj in subjects:
                layer = best_layers[(region, str(subj))]
                rprint(f"    {region} subj {subj}: {layer}", style="info")

        # 2. Build config, load model + neural data
        cfg = build_cfg(seed, neural_dataset)
        model = load_model(cfg, dev)
        model = configure_feature_extractor(cfg, model)

        loader_fn = load_all_nsd_data if neural_dataset == "nsd" else load_all_tvsd_data
        all_data = loader_fn(cfg, subjects=subjects, regions=regions)
        stimuli = all_data["stimuli"]
        shared_test_ids = all_data["shared_test_ids"]
        neural = all_data["neural"]
        rprint(f"  {len(shared_test_ids)} shared test stimuli\n", style="success")

        # 3. Build test-only dataloader
        test_stimuli = {sid: stimuli[sid] for sid in shared_test_ids if sid in stimuli}
        dl_test = _make_loader(
            test_stimuli, get_transform(ds_stats="imgnet"),
            cfg.batchsize, cfg.num_workers,
        )

        # 4. Re-extract unique best layers ONCE (no SRP)
        #    Filter to only layers needed for our regions/subjects
        #    (best_layers may include extra fine-grained regions from the DB)
        needed_layers = {
            best_layers[(region, str(subj))]
            for region in regions for subj in subjects
        }
        unique_layers = sorted(needed_layers)
        raw_acts = {}
        for layer in unique_layers:
            acts, _ = extract_single_layer(
                model, dl_test, dev, layer, shared_test_ids
            )
            raw_acts[layer] = acts

        del model, dl_test
        torch.cuda.empty_cache()

        # 5. Pre-compute neural RDMs (invariant across pca_k)
        neural_rdms = {}
        for region in regions:
            neural_rdms[region] = {}
            for subj in subjects:
                test_neural = neural[region][subj]["test"]
                responses = [
                    test_neural[sid] for sid in shared_test_ids
                    if sid in test_neural
                ]
                neural_tensor = torch.as_tensor(
                    np.stack(responses).squeeze(), dtype=torch.float32
                )
                neural_rdms[region][subj] = compute_rdm(neural_tensor)

        # 6. Sweep pca_k
        for pca_k in PCA_K_RANGE:
            rprint(f"\n  --- pca_k = {pca_k} ---", style="info")

            # Reconstruct each unique layer from k PCs
            reconstructed = {
                layer: reconstruct_from_pcs({layer: acts}, pca_k)[layer]
                for layer, acts in raw_acts.items()
            }

            # Build model RDMs
            model_rdms = {
                layer: compute_rdm(
                    acts.flatten(start_dim=1) if acts.ndim > 2 else acts
                )
                for layer, acts in reconstructed.items()
            }
            del reconstructed

            for region in regions:
                for subj in subjects:
                    best_layer = best_layers[(region, str(subj))]
                    score, ci_low, ci_high, boot_scores = bootstrap_rdm_correlation(
                        model_rdms[best_layer],
                        neural_rdms[region][subj],
                        COMPARE_METHOD,
                        n_bootstrap=cfg.n_bootstrap,
                    )

                    rprint(
                        f"    {region} subj {subj} | {best_layer} = {score:.4f}"
                        f"  [{ci_low:.4f}, {ci_high:.4f}]",
                        style="highlight",
                    )

                    cfg.pca_k = pca_k
                    cfg.region = region
                    cfg.subject_idx = subj
                    cfg.reconstruct_from_pcs = True
                    _save(cfg, best_layer, score, ci_low, ci_high, boot_scores)

            del model_rdms

        del raw_acts, neural_rdms
        torch.cuda.empty_cache()
        rprint(f"\n  Seed {seed} complete.\n", style="success")


# ── THINGS ────────────────────────────────────────────────────────────────────

def run_things():
    """Run pca_k reconstruction sweep for THINGS behavioral."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for seed in SEEDS:
        rprint(f"\n{'='*60}", style="info")
        rprint(f"  THINGS-BEHAVIOR | seed {seed}", style="info")
        rprint(f"{'='*60}\n", style="info")

        # 1. Query DB for best layer
        best_layers = query_best_layers("things-behavior", seed)
        best_layer = best_layers[("N/A", "N/A")]
        rprint(f"  Best layer from DB: {best_layer}", style="success")

        # 2. Build config, load model + THINGS data
        cfg = build_cfg(seed, "things-behavior")
        cfg.region = "N/A"
        cfg.subject_idx = "N/A"
        model = load_model(cfg, dev)
        model = configure_feature_extractor(cfg, model)

        neural_data, dl = get_neural_loader(cfg)
        rprint(f"  THINGS data loaded", style="success")

        # 3. SRP activations -> concept averaging (establishes concept mapping)
        acts, ids = get_activations(model, dl, dev)
        all_concepts = prepare_concept_alignment(cfg, acts, neural_data, ids)
        del acts
        torch.cuda.empty_cache()

        # 4. Fixed 80/20 concept split (seed=42, same as original pipeline)
        rng = np.random.RandomState(42)
        n_concepts = all_concepts.neural.size(0)
        perm = rng.permutation(n_concepts)
        n_sel = int(n_concepts * 0.2)
        eval_idx = perm[n_sel:]

        evaluation = AlignmentData(
            activations={
                l: a[eval_idx] for l, a in all_concepts.activations.items()
            },
            neural=all_concepts.neural[eval_idx],
            stimulus_ids=[all_concepts.stimulus_ids[i] for i in eval_idx],
            concept_image_ids={
                all_concepts.stimulus_ids[i]: all_concepts.concept_image_ids[
                    all_concepts.stimulus_ids[i]
                ]
                for i in eval_idx
            },
        )
        del all_concepts
        rprint(f"  {len(eval_idx)} evaluation concepts\n", style="success")

        # 5. Re-extract best layer without SRP ONCE
        raw_acts, raw_ids = extract_single_layer(model, dl, dev, best_layer)
        del model
        torch.cuda.empty_cache()

        # 6. Pre-compute neural RDM (invariant across pca_k)
        neural_rdm = compute_rdm(evaluation.neural)

        # 7. Sweep pca_k
        for pca_k in PCA_K_RANGE:
            rprint(f"\n  --- pca_k = {pca_k} ---", style="info")

            # Reconstruct from k PCs
            recon = reconstruct_from_pcs({best_layer: raw_acts}, pca_k)[best_layer]

            # Concept-average the eval set
            eval_acts = _concept_average_exact(recon, raw_ids, evaluation)

            # Build model RDM
            flat = eval_acts.flatten(start_dim=1) if eval_acts.ndim > 2 else eval_acts
            model_rdm = compute_rdm(flat)

            score, ci_low, ci_high, boot_scores = bootstrap_rdm_correlation(
                model_rdm, neural_rdm, COMPARE_METHOD, n_bootstrap=cfg.n_bootstrap,
            )

            rprint(
                f"    {best_layer} = {score:.4f}  [{ci_low:.4f}, {ci_high:.4f}]",
                style="highlight",
            )

            cfg.pca_k = pca_k
            cfg.reconstruct_from_pcs = True
            _save(cfg, best_layer, score, ci_low, ci_high, boot_scores)

        del raw_acts, neural_rdm
        torch.cuda.empty_cache()
        rprint(f"\n  Seed {seed} complete.\n", style="success")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Optimized reconstruction analysis: sweep pca_k for 1000-way model"
    )
    parser.add_argument(
        "--datasets", nargs="*",
        default=["nsd", "tvsd", "things-behavior"],
        choices=["nsd", "tvsd", "things-behavior"],
        help="Which neural datasets to run (default: all three)",
    )
    args = parser.parse_args()

    for ds in args.datasets:
        rprint(f"\n{'#'*60}", style="info")
        rprint(f"  RECONSTRUCTION ANALYSIS: {ds.upper()}", style="info")
        rprint(f"{'#'*60}", style="info")

        if ds in ("nsd", "tvsd"):
            run_nsd_tvsd(ds)
        else:
            run_things()


if __name__ == "__main__":
    main()
