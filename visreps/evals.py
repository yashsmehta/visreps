import torch
import pandas as pd
from omegaconf import OmegaConf, ListConfig
from visreps.utils import rprint, save_results
from visreps.utils import get_seed_letter
import visreps.models.utils as mutils
from visreps.dataloaders.neural import (
    get_neural_loader,
    load_all_nsd_data,
    load_all_tvsd_data,
    _make_loader,
)
from visreps.dataloaders.obj_cls import get_transform
from visreps.analysis.alignment import (
    AlignmentData,
    compute_traintest_alignment,
    prepare_traintest_alignment,
    prepare_concept_alignment,
    _align_stimulus_level,
)
from visreps.analysis.rsa import _concept_average_exact
from visreps.analysis.rsa import compute_rdm, compute_rdm_correlation
from visreps.analysis.reconstruct_from_pcs import reconstruct_from_pcs
import numpy as np


# ──────────────────────── helpers ────────────────────────
def _load_cfg(cfg):
    """Merge runtime cfg with training cfg (drops `mode`)."""
    seed_letter = get_seed_letter(cfg.seed)
    path = f"{cfg.checkpoint_dir}/cfg{cfg.cfg_id}{seed_letter}/config.json"
    base = OmegaConf.load(path)
    epoch = int(cfg.checkpoint_model.split('_')[-1].split('.')[0])
    base.epoch = epoch
    for k in ("mode", "exp_name", "lr_scheduler", "n_classes"):
        base.pop(k, None)
    return OmegaConf.merge(base, cfg)


def _build_header(cfg):
    """Build a compact one-line summary header for eval output."""
    analysis = cfg.get("analysis", "rsa").upper()
    seed = cfg.get("seed", "?")
    seed_letter = get_seed_letter(seed) if isinstance(seed, int) else "?"
    cfg_id = cfg.get("cfg_id", "?")
    epoch = cfg.get("epoch", "?")
    neural_dataset = cfg.get("neural_dataset", "?").upper()
    region = cfg.get("region", "")
    subject_idx = cfg.get("subject_idx", "")

    parts = [f"{analysis} eval"]
    parts.append(f"cfg{cfg_id}{seed_letter} epoch {epoch}")
    if region and str(region).upper() != "N/A":
        parts.append(f"{neural_dataset} {region}")
    else:
        parts.append(neural_dataset)
    if subject_idx != "" and str(subject_idx).upper() != "N/A":
        parts.append(f"subj {subject_idx}")
    parts.append(f"seed {seed}")
    return " | ".join(parts)


def _listify(val):
    """Ensure val is a plain Python list (handles int, str, ListConfig, list)."""
    if isinstance(val, (list, ListConfig)):
        return list(val)
    return [val]


# ───────────────────────── eval ──────────────────────────
def eval(cfg):
    """Unified evaluation: one forward pass, per-subject per-region results.

    Accepts list-valued cfg.subject_idx and cfg.region. For NSD/TVSD, loads all
    neural data once, extracts activations once, then iterates over all
    (subject, region) pairs internally.
    """
    verbose = cfg.get("verbose", False)

    # ── CONFIG & DEVICE ─────────────────────────────────
    if cfg.load_model_from == "checkpoint":
        cfg = _load_cfg(cfg)
    elif cfg.load_model_from == "torchvision":
        cfg.epoch = -1
        cfg.cfg_id = "pretrained" if cfg.pretrained_dataset == "imagenet1k" else "untrained"
        cfg.return_nodes = mutils.TORCHVISION_RETURN_NODES[cfg.model_name]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = cfg.neural_dataset.lower()

    # ── THINGS-BEHAVIOR: 80/20 concept-level train/test RSA ──
    if dataset == "things-behavior":
        rprint(f"\n  {_build_header(cfg)}\n", style="info")
        model = mutils.load_model(cfg, dev, verbose=verbose)
        model = mutils.configure_feature_extractor(cfg, model, verbose=verbose)

        neural_data, dl = get_neural_loader(cfg)
        rprint(f"  THINGS data loaded", style="success")

        acts, ids = mutils.get_activations(model, dl, dev)

        # Merge train/test images, average activations per concept
        all_concepts = prepare_concept_alignment(cfg, acts, neural_data, ids)
        del acts, neural_data, ids
        torch.cuda.empty_cache()

        # Fixed 80/20 split: 20% for layer selection, 80% for evaluation
        rng = np.random.RandomState(42)
        n_concepts = all_concepts.neural.size(0)
        perm = rng.permutation(n_concepts)
        n_sel = int(n_concepts * 0.2)
        sel_idx, eval_idx = perm[:n_sel], perm[n_sel:]

        selection = AlignmentData(
            activations={l: a[sel_idx] for l, a in all_concepts.activations.items()},
            neural=all_concepts.neural[sel_idx],
            stimulus_ids=[all_concepts.stimulus_ids[i] for i in sel_idx],
        )
        evaluation = AlignmentData(
            activations={l: a[eval_idx] for l, a in all_concepts.activations.items()},
            neural=all_concepts.neural[eval_idx],
            stimulus_ids=[all_concepts.stimulus_ids[i] for i in eval_idx],
            concept_image_ids={
                all_concepts.stimulus_ids[i]: all_concepts.concept_image_ids[all_concepts.stimulus_ids[i]]
                for i in eval_idx
            },
        )
        del all_concepts

        rprint(
            f"  {n_sel} selection concepts, {len(eval_idx)} evaluation concepts",
            style="success",
        )

        # Re-extract: get all images without SRP, concept-average for eval concepts
        def re_extract_fn(layer, sids=None):
            raw_acts, raw_ids = mutils.extract_single_layer(model, dl, dev, layer)
            if cfg.get("reconstruct_from_pcs"):
                raw_acts = reconstruct_from_pcs({layer: raw_acts}, cfg.pca_k)[layer]
                rprint(f"    Reconstructed from {cfg.pca_k} PCs", style="info")
            return _concept_average_exact(raw_acts, raw_ids, evaluation), evaluation.stimulus_ids

        alignment_scores = compute_traintest_alignment(
            cfg, selection, evaluation, verbose=verbose, re_extract_fn=re_extract_fn)

        del model, dl
        torch.cuda.empty_cache()

        results = pd.DataFrame(alignment_scores)
        if cfg.get("log_expdata"):
            save_results(results, cfg)
        return results

    # ── NSD / TVSD: unified multi-subject path ──────────
    subjects = _listify(cfg.subject_idx)
    regions = _listify(cfg.region)

    seed_letter = get_seed_letter(cfg.seed) if isinstance(cfg.seed, int) else "?"
    rprint(
        f"\n  {cfg.get('analysis', 'rsa').upper()} eval | cfg{cfg.get('cfg_id', '?')}{seed_letter} "
        f"epoch {cfg.get('epoch', '?')} | {cfg.neural_dataset.upper()} | "
        f"{len(subjects)} subjects x {len(regions)} regions | seed {cfg.seed}\n",
        style="info",
    )

    # Load model once
    model = mutils.load_model(cfg, dev, verbose=verbose)
    model = mutils.configure_feature_extractor(cfg, model, verbose=verbose)

    # Load all neural data once
    if dataset == "nsd":
        all_data = load_all_nsd_data(cfg, subjects=subjects, regions=regions)
    elif dataset == "tvsd":
        all_data = load_all_tvsd_data(cfg, subjects=subjects, regions=regions)
    else:
        raise ValueError(f"Unsupported neural_dataset='{dataset}' for multi-subject eval")

    stimuli = all_data["stimuli"]
    rprint(
        f"  {len(subjects)} subjects x {len(regions)} regions, "
        f"{len(stimuli)} stimuli, {len(all_data['shared_test_ids'])} shared test IDs",
        style="success",
    )

    # Single forward pass -> SRP activations
    transform = get_transform(ds_stats="imgnet")
    dl = _make_loader(stimuli, transform, cfg.batchsize, cfg.num_workers)
    acts, ids = mutils.get_activations(model, dl, dev)
    rprint(f"  Activations extracted once for all subjects/regions", style="success")
    del dl

    # Dispatch to analysis-specific helper
    analysis = cfg.get("analysis", "rsa").lower()
    if analysis == "rsa":
        results = _eval_rsa(cfg, model, acts, ids, all_data, subjects, regions, dev, verbose)
    elif analysis == "encoding_score":
        results = _eval_encoding(cfg, model, acts, ids, all_data, subjects, regions, verbose)
    else:
        raise ValueError(f"Unknown analysis method: {analysis}")

    torch.cuda.empty_cache()
    return results


# ──────────────────── RSA helper ────────────────────────
def _eval_rsa(cfg, model, acts, ids, all_data, subjects, regions, dev, verbose):
    """Two-phase RSA: layer selection with SRP, then re-extract without SRP.

    Phase 1: Per-(region, subject) layer selection using SRP activations.
    Phase 2: Re-extract unique best layers without SRP for exact test RDMs,
             then compute per-subject point estimates and optional bootstrap CIs.
    """
    method = cfg.get("compare_method", "spearman").lower()
    bootstrap = cfg.get("bootstrap", False)
    n_bootstrap = cfg.get("n_bootstrap", 1000)

    dataset = cfg.neural_dataset.lower()
    # Subsample train stimuli for layer selection to keep RDM computation fast.
    # Default 1,000 for both NSD (~9k train) and TVSD (~22k train).
    n_select = cfg.get("n_select", 1000)

    neural = all_data["neural"]
    shared_test_ids = all_data["shared_test_ids"]
    stimuli = all_data["stimuli"]

    # ══════════════════════════════════════════════════════
    # PHASE 1: Layer selection (uses SRP acts)
    # ══════════════════════════════════════════════════════
    rprint("\n  Phase 1: Per-subject layer selection", style="info")
    per_region_layers = {}       # {region: {subj: best_layer}}
    per_region_scores = {}       # {region: {subj: [{"layer": ..., "score": ...}]}}

    for region in regions:
        per_region_layers[region] = {}
        per_region_scores[region] = {}
        for subj in subjects:
            subj_neural_train = neural[region][subj]["train"]
            train_acts, train_neural, _ = _align_stimulus_level(
                acts, subj_neural_train, ids
            )

            n_train_subj = train_neural.size(0)
            if n_select is not None and n_select < n_train_subj:
                rng_sel = np.random.RandomState(42)
                sel_idx = rng_sel.choice(n_train_subj, size=n_select, replace=False)
            else:
                sel_idx = np.arange(n_train_subj)
            neural_rdm_sel = compute_rdm(train_neural[sel_idx])

            best_layer, best_score = None, -float("inf")
            subj_scores = []
            for layer, layer_acts in train_acts.items():
                flat = layer_acts[sel_idx].flatten(start_dim=1) if layer_acts.ndim > 2 else layer_acts[sel_idx]
                layer_rdm = compute_rdm(flat)
                score = compute_rdm_correlation(layer_rdm, neural_rdm_sel, correlation=method.capitalize())
                subj_scores.append({"layer": layer, "score": score})
                if score > best_score:
                    best_score = score
                    best_layer = layer

            per_region_layers[region][subj] = best_layer
            per_region_scores[region][subj] = subj_scores

            if verbose:
                rprint(
                    f"    {region} subj {subj}: {best_layer} ({best_score:.4f}), "
                    f"{len(sel_idx)} stimuli for selection",
                    style="info",
                )

            del train_acts, train_neural

    # ══════════════════════════════════════════════════════
    # FREE BULK ACTIVATIONS
    # ══════════════════════════════════════════════════════
    del acts
    torch.cuda.empty_cache()
    rprint("  Freed bulk SRP activations", style="success")

    # ══════════════════════════════════════════════════════
    # PHASE 2: Re-extract unique layers for test stimuli
    # ══════════════════════════════════════════════════════
    rprint("\n  Phase 2: Test evaluation", style="info")

    # Small test-only dataloader
    test_stimuli = {sid: stimuli[sid] for sid in shared_test_ids if sid in stimuli}
    transform = get_transform(ds_stats="imgnet")
    dl_test = _make_loader(test_stimuli, transform, cfg.batchsize, cfg.num_workers)
    rprint(f"  Test dataloader: {len(test_stimuli)} stimuli", style="success")

    # Collect unique best layers across all (region, subject) pairs
    all_unique_layers = set()
    for region_layers in per_region_layers.values():
        all_unique_layers.update(region_layers.values())

    # Re-extract each unique layer without SRP -> build model RDMs
    pca_k = cfg.get("pca_k", 1)
    model_rdms = {}
    for layer in sorted(all_unique_layers):
        rprint(f"  Re-extracting {layer} without SRP...", style="info")
        exact_acts, _ = mutils.extract_single_layer(model, dl_test, dev, layer, shared_test_ids)
        if cfg.get("reconstruct_from_pcs"):
            exact_acts = reconstruct_from_pcs({layer: exact_acts}, pca_k)[layer]
            rprint(f"    Reconstructed from {pca_k} PCs", style="info")
        flat = exact_acts.flatten(start_dim=1) if exact_acts.ndim > 2 else exact_acts
        model_rdms[layer] = compute_rdm(flat)
        del exact_acts

    # Free model (no longer needed)
    del model, dl_test
    torch.cuda.empty_cache()

    # ══════════════════════════════════════════════════════
    # Per-(region, subject) scoring + save
    # ══════════════════════════════════════════════════════
    all_results = []
    for region in regions:
        rprint(f"\n  -- Region: {region} --", style="info")

        for subj in subjects:
            best_layer = per_region_layers[region][subj]
            selection_scores = per_region_scores[region][subj]

            # Build neural RDM for this subject's test responses
            test_neural_dict = neural[region][subj]["test"]
            responses = [test_neural_dict[sid] for sid in shared_test_ids if sid in test_neural_dict]
            neural_tensor = torch.as_tensor(np.stack(responses).squeeze(), dtype=torch.float32)
            neural_rdm = compute_rdm(neural_tensor)

            # Point estimate
            point_estimate = compute_rdm_correlation(
                model_rdms[best_layer], neural_rdm, correlation=method.capitalize()
            )

            # Bootstrap (optional)
            ci_low, ci_high = None, None
            bootstrap_scores_list = None

            if bootstrap:
                rng = np.random.RandomState(42)
                n_test = neural_rdm.size(0)
                n_sub = int(n_test * 0.9)
                bootstrap_scores = np.empty(n_bootstrap, dtype=np.float64)

                for i in range(n_bootstrap):
                    idx = torch.from_numpy(
                        rng.choice(n_test, size=n_sub, replace=False)
                    ).to(neural_rdm.device)
                    m = model_rdms[best_layer][idx][:, idx]
                    n = neural_rdm[idx][:, idx]
                    bootstrap_scores[i] = compute_rdm_correlation(
                        m, n, correlation=method.capitalize()
                    )

                ci_low = float(np.percentile(bootstrap_scores, 2.5))
                ci_high = float(np.percentile(bootstrap_scores, 97.5))
                bootstrap_scores_list = bootstrap_scores.tolist()

            msg = f"    subj {subj} | {method.capitalize():<10}| {best_layer} = {point_estimate:.4f}"
            if bootstrap:
                msg += f"  [95% CI: {ci_low:.4f}, {ci_high:.4f}]"
            rprint(msg, style="highlight")

            result = {
                "layer": best_layer,
                "compare_method": method,
                "score": point_estimate,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "analysis": "rsa",
                "layer_selection_scores": selection_scores,
            }
            if bootstrap_scores_list is not None:
                result["bootstrap_scores"] = bootstrap_scores_list

            results_df = pd.DataFrame([result])

            if cfg.get("log_expdata"):
                save_cfg = OmegaConf.merge(cfg, {"subject_idx": subj, "region": region})
                save_results(results_df, save_cfg)

            all_results.append(result)

    return pd.DataFrame(all_results)


# ──────────────── encoding score helper ─────────────────
def _eval_encoding(cfg, model, acts, ids, all_data, subjects, regions, verbose):
    """Per-(region, subject) encoding score using SRP activations.

    Unlike RSA, encoding score uses SRP throughout (no re-extraction needed).
    """
    neural = all_data["neural"]

    all_results = []
    for region in regions:
        rprint(f"\n  -- Region: {region} --", style="info")

        for subj in subjects:
            subj_neural = neural[region][subj]

            train_data, test_data = prepare_traintest_alignment(
                cfg, acts, subj_neural, ids
            )

            alignment_scores = compute_traintest_alignment(
                cfg, train_data, test_data, verbose=verbose, re_extract_fn=None
            )

            # Free per-subject alignment data
            del train_data, test_data
            torch.cuda.empty_cache()

            results_df = pd.DataFrame(alignment_scores)

            if cfg.get("log_expdata"):
                save_cfg = OmegaConf.merge(cfg, {"subject_idx": subj, "region": region})
                save_results(results_df, save_cfg)

            all_results.extend(alignment_scores)

    # Free bulk activations and model
    del acts, model
    torch.cuda.empty_cache()

    return pd.DataFrame(all_results)
