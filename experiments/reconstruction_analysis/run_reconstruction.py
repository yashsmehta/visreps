"""
Reconstruction Analysis — Coarse-Layer Anchored
=================================================
For a coarse-grained model, finds the best layer per (region, subject). Then,
at that SAME layer, sweeps pca_k to measure how many PCs are needed to
reconstruct the alignment signal — for both the 1000-way and coarse models.

This answers: "At the layer where the coarse model peaks, how does the
alignment of each model's top-k PCs compare?"

Usage:
    python experiments/reconstruction_analysis/run_reconstruction.py --model-type coarse
    python experiments/reconstruction_analysis/run_reconstruction.py --model-type both --datasets nsd tvsd
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
FINE_CONFIG = {"cfg_id": 1000, "checkpoint_dir": "/data/ymehta3/default"}
SEEDS = [1, 2, 3]
COMPARE_METHOD = "spearman"

# Per-region coarse model configs for layer selection: region -> (cfg_id, checkpoint_dir)
COARSE_CONFIG = {
    "nsd": {
        "early visual stream": (64, "/data/ymehta3/alexnet_pca"),
        "ventral visual stream": (16, "/data/ymehta3/clip_pca"),
    },
    "tvsd": {
        "V1": (64, "/data/ymehta3/alexnet_pca"),
        "V4": (64, "/data/ymehta3/alexnet_pca"),
        "IT": (64, "/data/ymehta3/alexnet_pca"),
    },
    "things-behavior": {
        "N/A": (64, "/data/ymehta3/vit_pca"),
    },
}

# Dense at small k (1-10), increasingly sparse up to 50
PCA_K_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 40, 50]

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

def query_best_layers(neural_dataset, seed, cfg_id, checkpoint_dir, region=None):
    """Query results.db for a coarse model's best layers.

    Finds the best layer per (region, subject_idx) for the given coarse model.
    These layers are then used as anchors in the 1000-way model for PC
    reconstruction.

    If region is specified, only returns layers for that region.
    Returns dict mapping (region, subject_idx_str) -> best_layer_name.
    """
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT region, subject_idx, layer, score
        FROM results
        WHERE cfg_id = ?
          AND checkpoint_dir = ?
          AND reconstruct_from_pcs = 0
          AND analysis = 'rsa'
          AND compare_method = ?
          AND neural_dataset = ?
          AND seed = ?
    """
    params = [cfg_id, checkpoint_dir, COMPARE_METHOD, neural_dataset, seed]
    if region is not None:
        query += "  AND region = ?\n"
        params.append(region)
    df = pd.read_sql(query, conn, params=params)
    conn.close()

    if df.empty:
        raise ValueError(
            f"No baseline results for cfg_id={cfg_id} checkpoint_dir={checkpoint_dir} "
            f"{neural_dataset} seed={seed} region={region}. "
            "Run the coarse model evaluation first."
        )

    # Best layer per (region, subject_idx) — highest score
    idx = df.groupby(["region", "subject_idx"])["score"].idxmax()
    best = df.loc[idx]
    return {(row.region, row.subject_idx): row.layer for _, row in best.iterrows()}


# ── Config ────────────────────────────────────────────────────────────────────

def build_cfg(seed, neural_dataset, cfg_id, checkpoint_dir):
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
        "cfg_id": cfg_id,
        "checkpoint_dir": checkpoint_dir,
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


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _extract_at_anchor_layers(cfg, best_layers, regions, subjects,
                               dl_test, shared_test_ids, dev):
    """Load a model and extract activations at anchor layers (no SRP)."""
    model = load_model(cfg, dev)
    model = configure_feature_extractor(cfg, model)

    needed_layers = {
        best_layers[(region, str(subj))]
        for region in regions for subj in subjects
    }
    raw_acts = {}
    for layer in sorted(needed_layers):
        acts, _ = extract_single_layer(
            model, dl_test, dev, layer, shared_test_ids
        )
        raw_acts[layer] = acts

    del model
    torch.cuda.empty_cache()
    return raw_acts


def _sweep_pca_k(raw_acts, neural_rdms, best_layers, cfg, regions, subjects):
    """Sweep pca_k for one model's activations, compute RSA, and save."""
    for pca_k in PCA_K_VALUES:
        rprint(f"\n  --- pca_k = {pca_k} ---", style="info")

        reconstructed = reconstruct_from_pcs(raw_acts, pca_k)
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


# ── NSD / TVSD ────────────────────────────────────────────────────────────────

def run_nsd_tvsd(neural_dataset, model_types):
    """Run pca_k reconstruction sweep for NSD or TVSD."""
    ds = DATASET_CONFIG[neural_dataset]
    regions, subjects = ds["regions"], ds["subjects"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for seed in SEEDS:
        rprint(f"\n{'='*60}", style="info")
        rprint(f"  {neural_dataset.upper()} | seed {seed}", style="info")
        rprint(f"{'='*60}\n", style="info")

        # 1. Query DB for coarse model best layers per region (used as anchors)
        coarse_cfg = COARSE_CONFIG[neural_dataset]
        best_layers = {}
        rprint(f"  Coarse model best layers (per-region anchors):", style="success")
        for region in regions:
            cfg_id_r, ckpt_r = coarse_cfg[region]
            region_layers = query_best_layers(
                neural_dataset, seed, cfg_id_r, ckpt_r, region=region,
            )
            best_layers.update(region_layers)
            for subj in subjects:
                layer = best_layers[(region, str(subj))]
                rprint(f"    {region} subj {subj}: {layer}  "
                       f"(cfg_id={cfg_id_r}, {ckpt_r.split('/')[-1]})",
                       style="info")

        # 2. Load neural data (shared across model types)
        base_cfg = build_cfg(seed, neural_dataset,
                             FINE_CONFIG["cfg_id"], FINE_CONFIG["checkpoint_dir"])
        loader_fn = load_all_nsd_data if neural_dataset == "nsd" else load_all_tvsd_data
        all_data = loader_fn(base_cfg, subjects=subjects, regions=regions)
        stimuli = all_data["stimuli"]
        shared_test_ids = all_data["shared_test_ids"]
        neural = all_data["neural"]
        rprint(f"  {len(shared_test_ids)} shared test stimuli\n", style="success")

        # 3. Build test-only dataloader
        test_stimuli = {sid: stimuli[sid] for sid in shared_test_ids if sid in stimuli}
        dl_test = _make_loader(
            test_stimuli, get_transform(ds_stats="imgnet"),
            base_cfg.batchsize, base_cfg.num_workers,
        )

        # 4. Pre-compute neural RDMs (invariant across models and pca_k)
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

        # 5. Sweep per model type
        if "1000way" in model_types:
            rprint(f"\n  ── 1000-way model sweep ──", style="success")
            raw_acts = _extract_at_anchor_layers(
                base_cfg, best_layers, regions, subjects,
                dl_test, shared_test_ids, dev,
            )
            _sweep_pca_k(raw_acts, neural_rdms, best_layers, base_cfg,
                         regions, subjects)
            del raw_acts
            torch.cuda.empty_cache()

        if "coarse" in model_types:
            # Group regions by their coarse model to minimize model loads
            model_to_regions = {}
            for region in regions:
                key = coarse_cfg[region]
                model_to_regions.setdefault(key, []).append(region)

            for (cfg_id_c, ckpt_c), c_regions in model_to_regions.items():
                rprint(f"\n  ── Coarse model sweep: cfg_id={cfg_id_c}, "
                       f"{ckpt_c.split('/')[-1]} ──", style="success")
                cfg_c = build_cfg(seed, neural_dataset, cfg_id_c, ckpt_c)
                raw_acts = _extract_at_anchor_layers(
                    cfg_c, best_layers, c_regions, subjects,
                    dl_test, shared_test_ids, dev,
                )
                _sweep_pca_k(raw_acts, neural_rdms, best_layers, cfg_c,
                             c_regions, subjects)
                del raw_acts
                torch.cuda.empty_cache()

        del neural_rdms, dl_test
        torch.cuda.empty_cache()
        rprint(f"\n  Seed {seed} complete.\n", style="success")


# ── THINGS ────────────────────────────────────────────────────────────────────

def run_things(model_types):
    """Run pca_k reconstruction sweep for THINGS behavioral."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for seed in SEEDS:
        rprint(f"\n{'='*60}", style="info")
        rprint(f"  THINGS-BEHAVIOR | seed {seed}", style="info")
        rprint(f"{'='*60}\n", style="info")

        # 1. Query DB for coarse model's best layer (used as anchor)
        cfg_id_t, ckpt_t = COARSE_CONFIG["things-behavior"]["N/A"]
        best_layers = query_best_layers(
            "things-behavior", seed, cfg_id_t, ckpt_t,
        )
        best_layer = best_layers[("N/A", "N/A")]
        rprint(f"  Anchor layer: {best_layer}  "
               f"(cfg_id={cfg_id_t}, {ckpt_t.split('/')[-1]})", style="success")

        # 2. Establish concept mapping (model-independent, use first available)
        if "1000way" in model_types:
            ref_cfg_id, ref_ckpt = FINE_CONFIG["cfg_id"], FINE_CONFIG["checkpoint_dir"]
        else:
            ref_cfg_id, ref_ckpt = cfg_id_t, ckpt_t

        ref_cfg = build_cfg(seed, "things-behavior", ref_cfg_id, ref_ckpt)
        ref_cfg.region = "N/A"
        ref_cfg.subject_idx = "N/A"

        model = load_model(ref_cfg, dev)
        model = configure_feature_extractor(ref_cfg, model)
        neural_data, dl = get_neural_loader(ref_cfg)
        rprint(f"  THINGS data loaded", style="success")

        acts, ids = get_activations(model, dl, dev)
        all_concepts = prepare_concept_alignment(ref_cfg, acts, neural_data, ids)
        del acts
        torch.cuda.empty_cache()

        # Keep reference model alive for reuse in the first sweep
        ref_model = model

        # 3. Fixed 80/20 concept split (seed=42, same as original pipeline)
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

        # 4. Pre-compute neural RDM (shared across model types)
        neural_rdm = compute_rdm(evaluation.neural)
        evaluation.activations.clear()  # not used in sweep; free memory

        # 5. Define models to sweep
        models_to_run = []
        if "1000way" in model_types:
            models_to_run.append(
                ("1000-way model", FINE_CONFIG["cfg_id"], FINE_CONFIG["checkpoint_dir"]))
        if "coarse" in model_types:
            models_to_run.append(
                (f"Coarse model (cfg_id={cfg_id_t})", cfg_id_t, ckpt_t))

        # 6. Sweep each model
        for label, m_cfg_id, m_ckpt in models_to_run:
            rprint(f"\n  ── {label} sweep ──", style="success")
            cfg = build_cfg(seed, "things-behavior", m_cfg_id, m_ckpt)
            cfg.region = "N/A"
            cfg.subject_idx = "N/A"

            # Reuse reference model if it matches, otherwise load fresh
            if ref_model is not None and m_cfg_id == ref_cfg_id and m_ckpt == ref_ckpt:
                mdl = ref_model
                ref_model = None  # consumed
            else:
                mdl = load_model(cfg, dev)
                mdl = configure_feature_extractor(cfg, mdl)
            raw_acts, raw_ids = extract_single_layer(mdl, dl, dev, best_layer)
            del mdl
            torch.cuda.empty_cache()

            for pca_k in PCA_K_VALUES:
                rprint(f"\n  --- pca_k = {pca_k} ---", style="info")
                recon = reconstruct_from_pcs({best_layer: raw_acts}, pca_k)[best_layer]
                eval_acts = _concept_average_exact(recon, raw_ids, evaluation)
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

            del raw_acts
            torch.cuda.empty_cache()

        del ref_model, neural_rdm
        torch.cuda.empty_cache()
        rprint(f"\n  Seed {seed} complete.\n", style="success")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Reconstruction analysis: sweep pca_k for 1000-way and/or coarse models"
    )
    parser.add_argument(
        "--datasets", nargs="*",
        default=["nsd", "tvsd", "things-behavior"],
        choices=["nsd", "tvsd", "things-behavior"],
        help="Which neural datasets to run (default: all three)",
    )
    parser.add_argument(
        "--model-type", default="coarse",
        choices=["1000way", "coarse", "both"],
        help="Which model(s) to sweep (default: coarse)",
    )
    args = parser.parse_args()

    model_types = (["1000way", "coarse"] if args.model_type == "both"
                   else [args.model_type])

    for ds in args.datasets:
        rprint(f"\n{'#'*60}", style="info")
        rprint(f"  RECONSTRUCTION ANALYSIS: {ds.upper()}", style="info")
        rprint(f"{'#'*60}", style="info")

        if ds in ("nsd", "tvsd"):
            run_nsd_tvsd(ds, model_types)
        else:
            run_things(model_types)


if __name__ == "__main__":
    main()
