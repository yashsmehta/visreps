import os
import pickle
import sqlite3

import torch
import pandas as pd
from omegaconf import OmegaConf, ListConfig
from visreps.utils import rprint, save_results, _compute_run_id
from visreps.utils import get_seed_letter
import visreps.models.utils as mutils
from visreps.dataloaders.neural import (
    get_neural_loader,
    load_all_nsd_data,
    load_nsd_synthetic_test_data,
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
from visreps.analysis.rsa import compute_rdm, compute_rdm_correlation, score_rdm_pair
from visreps.analysis.encoding_score import select_layer_scores, evaluate_layer
from visreps.analysis.reconstruct_from_pcs import reconstruct_from_pcs
import numpy as np


# ──────────────────────── helpers ────────────────────────
def _load_cfg(cfg):
    """Merge runtime cfg with training cfg (drops `mode`)."""
    seed_letter = get_seed_letter(cfg.seed)
    path = f"{cfg.checkpoint_dir}/cfg{cfg.cfg_id}{seed_letter}/config.json"
    base = OmegaConf.load(path)
    # Parse epoch from filename: checkpoint_epoch_100.pth or checkpoint_epoch_100_recal.pth
    parts = cfg.checkpoint_model.replace('.pth', '').split('_')
    epoch = int(next(p for p in reversed(parts) if p.isdigit()))
    base.epoch = epoch
    for k in ("mode", "exp_name", "lr_scheduler", "n_classes"):
        base.pop(k, None)
    return OmegaConf.merge(base, cfg)


def _print_header(cfg, n_subjects=None, n_regions=None):
    """Print a styled header block for eval output."""
    analysis = cfg.get("analysis", "rsa").upper()
    method = cfg.get("compare_method", "spearman").capitalize()
    seed = cfg.get("seed", "?")
    seed_letter = get_seed_letter(seed) if isinstance(seed, int) else "?"
    cfg_id = cfg.get("cfg_id", "?")
    epoch = cfg.get("epoch", "?")
    neural_dataset = cfg.get("neural_dataset", "?").upper()

    title = f"{analysis} · {method}"
    rprint(f"\n  ── {title} {'─' * max(1, 42 - len(title))}", style="highlight")
    # Use dot separator for word-like cfg_ids (e.g. "pretrained"), no separator for numeric
    cfg_label = f"cfg{cfg_id}.{seed_letter}" if not str(cfg_id).isdigit() else f"cfg{cfg_id}{seed_letter}"
    rprint(f"  {cfg_label} · seed {seed} · epoch {epoch}", style="info")

    parts = [neural_dataset]
    if n_subjects is not None and n_regions is not None:
        parts.append(f"{n_subjects} subjects × {n_regions} regions")
    rprint(f"  {' · '.join(parts)}", style="info")
    rprint("")


def _print_region_results(region, layer, scores, subjects, ci_lows=None, ci_highs=None):
    """Print per-subject scores for a region (one shared layer) with mean summary."""
    rprint(f"\n  ── {region} · {layer} {'─' * max(1, 40 - len(region) - len(layer))}", style="info")
    for i, subj in enumerate(subjects):
        msg = f"    S{subj:<3} {scores[i]:.4f}"
        if ci_lows is not None:
            msg += f"  [not bold grey50]\\[{ci_lows[i]:.4f}, {ci_highs[i]:.4f}][/not bold grey50]"
        rprint(msg, style="highlight")
    rprint(f"    {'─' * 34}", style="info")
    rprint(f"    Mean{' ' * 5}{np.mean(scores):.4f} ± {np.std(scores):.4f}", style="highlight")


def _print_cross_region_summary(region_means):
    """Print a final summary line for each region when multiple regions are evaluated."""
    if len(region_means) > 1:
        rprint(f"\n  {'═' * 46}", style="highlight")
        for region, mean in region_means.items():
            rprint(f"  {region:<30} {mean:.4f}", style="highlight")


def _listify(val):
    """Ensure val is a plain Python list (handles int, str, ListConfig, list)."""
    if isinstance(val, (list, ListConfig)):
        return list(val)
    return [val]


def _get_eval_transform(cfg):
    """Return the correct preprocessing transform based on model."""
    stats = "clip" if "CLIP" in cfg.get("model_name", "") else "imgnet"
    return get_transform(ds_stats=stats)


def _set_torchvision_cfg(cfg):
    """Set epoch and cfg_id for torchvision-loaded models."""
    cfg.epoch = -1
    cfg.cfg_id = "untrained" if cfg.get("pretrained_dataset", "none") == "none" else "pretrained"
    return cfg


# ──────────────── shared RSA helpers ─────────────────────
def _make_rsa_result(layer, method, score, ci_low, ci_high,
                     selection_scores, bootstrap_scores=None):
    """Build standardized RSA result dict."""
    result = {
        "layer": layer,
        "compare_method": method,
        "score": score,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "analysis": "rsa",
        "layer_selection_scores": selection_scores,
    }
    if bootstrap_scores is not None:
        result["bootstrap_scores"] = bootstrap_scores
    return result


def _best_layer_across_subjects(subject_scores):
    """Pick the single layer with the highest mean selection score across subjects.

    Args:
        subject_scores: {subj: [{layer, score}]} — same layers for every subject.
    """
    mean_scores = pd.DataFrame(
        {subj: {s["layer"]: s["score"] for s in scores} for subj, scores in subject_scores.items()}
    ).mean(axis=1)
    return mean_scores.idxmax(), float(mean_scores.max())


def _select_rsa_layers(acts, ids, neural, subjects, regions,
                       method, n_select=1000, verbose=False):
    """Per-region layer selection using SRP activations.

    Each subject scores every layer on its own train stimuli; the region's
    layer is the one with the highest mean score across subjects.

    Returns:
        per_region_layer: {region: best_layer_name}
        per_region_scores: {region: {subj: [{layer, score}]}}
    """
    per_region_layer = {}
    per_region_scores = {}

    for region in regions:
        per_region_scores[region] = {}
        for subj in subjects:
            train_acts, train_neural, _ = _align_stimulus_level(
                acts, neural[region][subj]["train"], ids
            )

            n_train = train_neural.size(0)
            if n_select is not None and n_select < n_train:
                sel_idx = np.random.RandomState(42).choice(
                    n_train, size=n_select, replace=False
                )
            else:
                sel_idx = np.arange(n_train)

            neural_rdm = compute_rdm(train_neural[sel_idx])

            scores = []
            for layer, layer_acts in train_acts.items():
                flat = (layer_acts[sel_idx].flatten(start_dim=1)
                        if layer_acts.ndim > 2 else layer_acts[sel_idx])
                score = compute_rdm_correlation(
                    compute_rdm(flat), neural_rdm,
                    correlation=method.capitalize(),
                )
                scores.append({"layer": layer, "score": score})
            per_region_scores[region][subj] = scores
            del train_acts, train_neural

        best_layer, best_score = _best_layer_across_subjects(per_region_scores[region])
        per_region_layer[region] = best_layer
        rprint(
            f"    {region}: {best_layer} (mean selection score {best_score:.4f} "
            f"across {len(subjects)} subjects)",
            style="info",
        )

    return per_region_layer, per_region_scores


def _reextract_and_score(model, cfg, dev, test_stimuli, test_ids,
                         test_neural, best_layers, regions, subjects,
                         selection_scores=None, verbose=False):
    """Re-extract unique best layers without SRP, score per (region, subject).

    Args:
        test_stimuli: {sid: image} for building test dataloader.
        test_ids: ordered list of test stimulus IDs.
        test_neural: {region: {subj: {sid: response}}} — test-only responses.
        best_layers: {region: layer_name} — one layer per region, shared by all subjects.
        selection_scores: {region: {subj: [{layer, score}]}} or None.

    Returns:
        pd.DataFrame with one row per (region, subject).
    """
    method = cfg.get("compare_method", "spearman").lower()
    bootstrap = cfg.get("bootstrap", False)
    n_bootstrap = cfg.get("n_bootstrap", 1000)
    pca_k = cfg.get("pca_k", 1)

    # Build test dataloader
    transform = _get_eval_transform(cfg)
    dl_test = _make_loader(test_stimuli, transform, cfg.batchsize, cfg.num_workers)
    rprint(f"  Test dataloader: {len(test_stimuli)} stimuli", style="success")

    # Re-extract unique best layers without SRP
    model_rdms = {}
    for layer in sorted(set(best_layers.values())):
        exact_acts, _ = mutils.extract_single_layer(
            model, dl_test, dev, layer, test_ids
        )
        if cfg.get("reconstruct_from_pcs"):
            exact_acts = reconstruct_from_pcs({layer: exact_acts}, pca_k)[layer]
            rprint(f"    Reconstructed from {pca_k} PCs", style="info")
        flat = exact_acts.flatten(start_dim=1) if exact_acts.ndim > 2 else exact_acts
        model_rdms[layer] = compute_rdm(flat)
        del exact_acts

    del model, dl_test
    torch.cuda.empty_cache()

    # Score per (region, subject)
    all_results = []
    region_means = {}
    for region in regions:
        best_layer = best_layers[region]
        region_scores, region_ci_lows, region_ci_highs = [], [], []
        for subj in subjects:
            # Build neural RDM
            responses = [
                test_neural[region][subj][sid]
                for sid in test_ids
                if sid in test_neural[region][subj]
            ]
            neural_rdm = compute_rdm(torch.as_tensor(np.stack(responses), dtype=torch.float32))

            # Score + bootstrap
            score, ci_low, ci_high, boot_scores = score_rdm_pair(
                model_rdms[best_layer], neural_rdm, method,
                bootstrap=bootstrap, n_bootstrap=n_bootstrap,
            )

            # Build and save result
            sel_scores = (selection_scores[region][subj]
                          if selection_scores else [])
            result = _make_rsa_result(
                best_layer, method, score, ci_low, ci_high,
                sel_scores, boot_scores,
            )
            result["region"] = region
            result["subject_idx"] = subj

            if cfg.get("log_expdata"):
                save_cfg = OmegaConf.merge(
                    cfg, {"subject_idx": subj, "region": region}
                )
                save_results(pd.DataFrame([result]), save_cfg, quiet=True)

            all_results.append(result)
            region_scores.append(score)
            region_ci_lows.append(ci_low)
            region_ci_highs.append(ci_high)

        _print_region_results(
            region, best_layer, region_scores, subjects,
            ci_lows=region_ci_lows if bootstrap else None,
            ci_highs=region_ci_highs if bootstrap else None,
        )
        if cfg.get("log_expdata"):
            rprint(f"    Saved {len(subjects)} results to results.db", style="success")
        region_means[region] = np.mean(region_scores)

    _print_cross_region_summary(region_means)
    return pd.DataFrame(all_results)


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
        cfg = _set_torchvision_cfg(cfg)
    cfg.return_nodes = list(mutils.TORCHVISION_RETURN_NODES[cfg.model_name])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = cfg.neural_dataset.lower()

    # ── THINGS-BEHAVIOR: 80/20 concept-level train/test RSA ──
    if dataset == "things-behavior":
        _print_header(cfg)
        model = mutils.load_model(cfg, dev, verbose=verbose)
        model = mutils.configure_feature_extractor(cfg, model, verbose=verbose)

        neural_data, dl = get_neural_loader(cfg)
        if "CLIP" in cfg.get("model_name", ""):
            dl.dataset.transform = _get_eval_transform(cfg)
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

        # Re-extract: concept-average on-the-fly to avoid materializing full tensor
        # Build reverse map: image_id -> list of concept names it belongs to
        _img_to_concepts = {}
        for concept in evaluation.stimulus_ids:
            for img_id in evaluation.concept_image_ids[concept]:
                _img_to_concepts.setdefault(str(img_id), []).append(concept)

        def re_extract_fn(layer, sids=None):
            model.eval()
            concept_sums = {}   # concept -> running sum tensor
            concept_counts = {} # concept -> int
            with torch.no_grad():
                for imgs, keys in dl:
                    feats = model(imgs.to(dev))
                    out = feats[layer].reshape(feats[layer].size(0), -1).cpu().float()
                    if cfg.get("reconstruct_from_pcs"):
                        out = reconstruct_from_pcs({layer: out}, cfg.pca_k)[layer]
                    for i, key in enumerate(keys):
                        for concept in _img_to_concepts.get(str(key), []):
                            if concept not in concept_sums:
                                concept_sums[concept] = torch.zeros(out.size(1))
                                concept_counts[concept] = 0
                            concept_sums[concept] += out[i]
                            concept_counts[concept] += 1
            avgs = []
            for concept in evaluation.stimulus_ids:
                if concept in concept_sums and concept_counts[concept] > 0:
                    avgs.append(concept_sums[concept] / concept_counts[concept])
                else:
                    avgs.append(torch.zeros(next(iter(concept_sums.values())).size(0)))
            rprint(f"  ✓ Re-extracted {layer}: streaming concept-average ({len(avgs)} concepts)", style="success")
            return torch.stack(avgs), evaluation.stimulus_ids

        alignment_scores = compute_traintest_alignment(
            cfg, selection, evaluation, verbose=verbose, re_extract_fn=re_extract_fn)

        del model, dl
        torch.cuda.empty_cache()

        results = pd.DataFrame(alignment_scores)
        if cfg.get("log_expdata"):
            save_results(results, cfg)
        return results

    # ── NSD SYNTHETIC: dedicated RSA path (reuses NSD layer selection) ──
    if dataset == "nsd_synthetic":
        subjects = _listify(cfg.subject_idx)
        regions = _listify(cfg.region)
        _print_header(cfg, len(subjects), len(regions))
        return _eval_rsa_nsd_synthetic(cfg, subjects, regions, dev, verbose)

    # ── CUSACK 2025: dedicated RSA path (reuses NSD layer selection) ──
    if dataset == "cusack":
        age_groups = _listify(cfg.subject_idx)
        regions = _listify(cfg.region)
        _print_header(cfg, len(age_groups), len(regions))
        return _eval_rsa_cusack(cfg, age_groups, regions, dev, verbose)

    # ── NSD / TVSD: unified multi-subject path ──────────
    subjects = _listify(cfg.subject_idx)
    regions = _listify(cfg.region)

    _print_header(cfg, len(subjects), len(regions))

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
        f"  {len(stimuli)} stimuli, {len(all_data['shared_test_ids'])} shared test",
        style="success",
    )

    # Single forward pass -> SRP activations
    transform = _get_eval_transform(cfg)
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
    """Two-phase RSA: layer selection with SRP, then re-extract without SRP."""
    method = cfg.get("compare_method", "spearman").lower()
    n_select = cfg.get("n_select", 1000)
    neural = all_data["neural"]

    # Phase 1: layer selection
    rprint("\n  Phase 1: Per-subject layer selection", style="info")
    per_region_layers, per_region_scores = _select_rsa_layers(
        acts, ids, neural, subjects, regions, method, n_select, verbose
    )
    del acts
    torch.cuda.empty_cache()
    rprint("  Freed bulk SRP activations", style="success")

    # Phase 2: re-extract and score
    rprint("\n  Phase 2: Test evaluation", style="info")
    stimuli = all_data["stimuli"]
    shared_test_ids = all_data["shared_test_ids"]
    test_stimuli = {sid: stimuli[sid] for sid in shared_test_ids if sid in stimuli}
    test_neural = {r: {s: neural[r][s]["test"] for s in subjects} for r in regions}

    return _reextract_and_score(
        model, cfg, dev, test_stimuli, shared_test_ids,
        test_neural, per_region_layers, regions, subjects,
        per_region_scores, verbose,
    )


# ───────────── NSD Synthetic RSA helper ──────────────────
def _lookup_nsd_best_layers(cfg, subjects, regions):
    """Look up the per-region RSA layer from regular NSD evaluation results.

    Computes the run_id that the corresponding NSD eval would have produced for
    each subject and queries the results DB. All subjects of a region must have
    the same layer (one layer per ROI); otherwise the NSD eval is stale.

    Returns: {region: layer_name}
    """
    method = cfg.get("compare_method", "spearman").lower()
    conn = sqlite3.connect("results.db")

    layers = {}
    for region in regions:
        subject_layers = {}
        for subj in subjects:
            nsd_cfg = OmegaConf.merge(cfg, {
                "neural_dataset": "nsd",
                "analysis": "rsa",
                "subject_idx": subj,
                "region": region,
                "compare_method": method,
            })
            run_id = _compute_run_id(nsd_cfg)

            row = pd.read_sql_query(
                "SELECT layer FROM results WHERE run_id=? AND compare_method=?",
                conn, params=(run_id, method),
            )
            if row.empty:
                raise ValueError(
                    f"No NSD RSA result found (run_id={run_id}) for "
                    f"seed={cfg.seed}, region={region}, subj={subj}, "
                    f"cfg_id={cfg.cfg_id}. Run NSD eval first."
                )
            subject_layers[subj] = row.iloc[0]["layer"]

        if len(set(subject_layers.values())) != 1:
            raise ValueError(
                f"NSD RSA results for region={region} have different layers per subject "
                f"({subject_layers}). Re-run the NSD eval so one layer is selected per ROI."
            )
        layers[region] = next(iter(subject_layers.values()))

    conn.close()
    return layers


def _eval_rsa_nsd_synthetic(cfg, subjects, regions, dev, verbose):
    """RSA on NSD Synthetic: reuse best layers from NSD, score on synthetic stimuli."""
    best_layers = _lookup_nsd_best_layers(cfg, subjects, regions)
    for region, layer in best_layers.items():
        rprint(f"    {region}: reusing layer {layer} from NSD", style="info")

    test_data = load_nsd_synthetic_test_data(cfg, subjects=subjects, regions=regions)
    rprint(f"  Loaded {len(test_data['test_ids'])} synthetic test stimuli", style="success")

    model = mutils.load_model(cfg, dev, verbose=verbose)
    model = mutils.configure_feature_extractor(cfg, model, verbose=verbose)

    return _reextract_and_score(
        model, cfg, dev,
        test_data["stimuli"], test_data["test_ids"], test_data["neural"],
        best_layers, regions, subjects, verbose=verbose,
    )


# ──────────────── Cusack 2025 RSA helper ─────────────────
def _eval_rsa_cusack(cfg, age_groups, regions, dev, verbose):
    """RSA on Cusack: score all layers (no SRP) on 36 infant stimuli."""
    with open("datasets/neural/cusack2025/fmri_responses.pkl", "rb") as f:
        fmri = pickle.load(f)
    test_ids = sorted(fmri[regions[0]][age_groups[0]].keys())
    stimuli = {sid: os.path.join("datasets/neural/cusack2025/display_images", f"{sid}.png")
               for sid in test_ids}

    # One forward pass, no SRP — 36 stimuli is tiny
    model = mutils.load_model(cfg, dev, verbose=verbose)
    model = mutils.configure_feature_extractor(cfg, model, verbose=verbose)
    dl = _make_loader(stimuli, _get_eval_transform(cfg), cfg.batchsize, cfg.num_workers)

    model.eval()
    acts = {}
    with torch.no_grad():
        for imgs, keys in dl:
            features = model(imgs.to(dev))
            for layer, feat in features.items():
                acts.setdefault(layer, []).append(feat.cpu().flatten(start_dim=1))
    acts = {layer: torch.cat(t) for layer, t in acts.items()}
    rprint(f"  Extracted {len(acts)} layers × {len(test_ids)} stimuli (no SRP)", style="success")

    del model, dl
    torch.cuda.empty_cache()

    # Precompute model RDMs once (don't depend on region/age_group)
    model_rdms = {layer: compute_rdm(a) for layer, a in acts.items()}
    del acts

    method = cfg.get("compare_method", "spearman").lower()
    correlation = method.capitalize()
    all_results = []

    for region in regions:
        for ag in age_groups:
            responses = [fmri[region][ag][sid] for sid in test_ids]
            neural_rdm = compute_rdm(torch.as_tensor(np.stack(responses), dtype=torch.float32))

            layer_scores = [
                {"layer": layer, "score": compute_rdm_correlation(
                    rdm, neural_rdm, correlation=correlation)}
                for layer, rdm in model_rdms.items()
            ]
            best = max(layer_scores, key=lambda x: x["score"])

            result = _make_rsa_result(
                best["layer"], method, best["score"], None, None, layer_scores)
            result["region"] = region
            result["subject_idx"] = ag

            if cfg.get("log_expdata"):
                save_cfg = OmegaConf.merge(cfg, {"subject_idx": ag, "region": region})
                save_results(pd.DataFrame([result]), save_cfg, quiet=True)

            all_results.append(result)

            rprint(f"\n  ── {region} / {ag} {'─' * max(1, 30 - len(region) - len(ag))}", style="info")
            for entry in layer_scores:
                marker = " ◀" if entry["layer"] == best["layer"] else ""
                rprint(f"    {entry['layer']:12s} {entry['score']:.4f}{marker}", style="highlight")

    if cfg.get("log_expdata"):
        rprint(f"\n    Saved {len(all_results)} results to results.db", style="success")
    return pd.DataFrame(all_results)


# ──────────────── encoding score helper ─────────────────
def _eval_encoding(cfg, model, acts, ids, all_data, subjects, regions, verbose):
    """Per-region encoding score using SRP activations (no re-extraction).

    Phase 1: every subject scores every layer on an 80/20 split of its train
    data; the region's layer is the best on average across subjects.
    Phase 2: that layer is refit on each subject's full train data and scored
    on its test data.
    """
    neural = all_data["neural"]
    bootstrap = cfg.get("bootstrap", True)
    n_bootstrap = cfg.get("n_bootstrap", 1000)
    pca_k = cfg.get("pca_k", 1) if cfg.get("reconstruct_from_pcs") else None

    def _subject_data(region, subj):
        return prepare_traintest_alignment(cfg, acts, neural[region][subj], ids)

    all_results = []
    region_means = {}
    for region in regions:
        rprint(f"\n  Phase 1: layer selection for {region}", style="info")
        selection_scores = {}
        for subj in subjects:
            train_data, _ = _subject_data(region, subj)
            selection_scores[subj] = select_layer_scores(train_data, verbose=verbose)
            del train_data
        best_layer, best_score = _best_layer_across_subjects(selection_scores)
        rprint(
            f"    {region}: {best_layer} (mean val r {best_score:.4f} "
            f"across {len(subjects)} subjects)",
            style="info",
        )

        rprint(f"  Phase 2: test evaluation for {region}", style="info")
        region_scores = []
        for subj in subjects:
            train_data, test_data = _subject_data(region, subj)
            result = evaluate_layer(
                best_layer, train_data, test_data,
                bootstrap=bootstrap, n_bootstrap=n_bootstrap,
                verbose=verbose, reconstruct_pca_k=pca_k,
            )
            del train_data, test_data
            result["layer_selection_scores"] = selection_scores[subj]
            result["region"] = region
            result["subject_idx"] = subj

            if cfg.get("log_expdata"):
                save_cfg = OmegaConf.merge(cfg, {"subject_idx": subj, "region": region})
                save_results(pd.DataFrame([result]), save_cfg, quiet=True)

            all_results.append(result)
            region_scores.append(result["score"])

        _print_region_results(region, best_layer, region_scores, subjects)
        if cfg.get("log_expdata"):
            rprint(f"    Saved {len(subjects)} results to results.db", style="success")
        region_means[region] = np.mean(region_scores)

    _print_cross_region_summary(region_means)

    # Free bulk activations and model
    del acts, model
    torch.cuda.empty_cache()

    return pd.DataFrame(all_results)
