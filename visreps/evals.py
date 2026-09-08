import torch
import pandas as pd
from omegaconf import OmegaConf, ListConfig
import sqlite3
import visreps.utils as vutils
from visreps.utils import rprint, save_results
from visreps.utils import get_seed_letter
import visreps.models.utils as mutils
from visreps.models.batchnorm import prepare_eval_batchnorm, training_image_ids
from visreps.dataloaders.neural import (
    get_neural_loader,
    load_all_nsd_data,
    load_all_nsd_synthetic_data,
    load_all_tvsd_data,
    NsdSyntheticTransform,
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
    """Return the correct preprocessing transform based on dataset and model."""
    stats = "clip" if "CLIP" in cfg.get("model_name", "") else "imgnet"
    if cfg.get("neural_dataset", "").lower() == "nsd_synthetic":
        return NsdSyntheticTransform(ds_stats=stats)
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


_TVSD_SELECT_FRAC = 0.2


def _tvsd_train_split(all_data, *, select_frac=_TVSD_SELECT_FRAC,
                      n_report=None, seed=42):
    """Re-split TVSD so RSA is selected and reported on *train* stimuli.

    TVSD's held-out set is only 100 stimuli. This instead splits the 22,248
    train stimuli at the stimulus level: ``select_frac`` for layer selection and
    the remainder for reporting. Both monkeys saw the same images, so one split
    serves every subject and region and the per-region layer stays comparable.

    Note the train stimuli were each presented once, so the reporting RDM is
    built from single-trial responses -- scores are attenuated relative to the
    30-repetition test set, and no within-subject noise ceiling exists for them.

    The rewrite puts the selection half under "train" and the reporting half
    under "test", so every downstream RSA step runs unchanged.

    Returns:
        (select_ids, report_ids)
    """
    neural = all_data["neural"]
    regions, subjects = all_data["regions"], all_data["subjects"]

    common = sorted(set.intersection(*[
        set(neural[r][s]["train"]) for r in regions for s in subjects
    ]))
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(common))
    n_sel = int(round(select_frac * len(common)))
    select_ids = sorted(common[i] for i in order[:n_sel])
    report_ids = sorted(common[i] for i in order[n_sel:])

    if n_report is not None and n_report < len(report_ids):
        keep = rng.choice(len(report_ids), size=n_report, replace=False)
        report_ids = sorted(report_ids[i] for i in keep)

    for region in regions:
        for subj in subjects:
            train = neural[region][subj]["train"]
            neural[region][subj] = {
                "train": {sid: train[sid] for sid in select_ids},
                "test": {sid: train[sid] for sid in report_ids},
            }
    all_data["shared_test_ids"] = report_ids
    return select_ids, report_ids


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


def _lookup_nsd_best_layers(cfg, subjects, regions):
    """Reuse each ROI's layer from the matching regular-NSD RSA run in results.db.

    NSD-synthetic has no training split, so no layer is selected here: it comes
    from the regular-NSD run with the same model, seed and epoch. One layer per
    ROI, so every requested subject of a region must agree.
    """
    match = {
        "neural_dataset": "nsd",
        "analysis": "rsa",
        "compare_method": cfg.get("compare_method", "spearman").lower(),
        "seed": cfg.get("seed"),
        "epoch": cfg.get("epoch"),
        "cfg_id": cfg.get("cfg_id"),
        "model_name": cfg.get("model_name"),
        "checkpoint_dir": cfg.get("checkpoint_dir"),
        "reconstruct_from_pcs": bool(cfg.get("reconstruct_from_pcs", False)),
    }
    where = " AND ".join(f"{field} IS ?" for field in match)
    # Same database results are written to; read-only so a lookup can never write.
    conn = sqlite3.connect(f"file:{vutils._RESULTS_DB_PATH}?mode=ro", uri=True)
    try:
        rows = pd.read_sql_query(
            f"SELECT region, subject_idx, layer FROM results WHERE {where}",
            conn, params=list(match.values()))
    finally:
        conn.close()
    rows = rows[rows.subject_idx.astype(str).isin({str(s) for s in subjects})]

    layers = {}
    for region in regions:
        found = rows[rows.region == region]
        if found.empty:
            raise ValueError(
                f"No regular-NSD RSA result for region={region} with "
                f"{ {k: v for k, v in match.items() if k not in ('neural_dataset', 'analysis')} }. "
                f"Run the same eval with neural_dataset=nsd first.")
        if found.layer.nunique() != 1:
            raise ValueError(
                f"NSD results for region={region} disagree on the layer "
                f"({dict(zip(found.subject_idx, found.layer))}). Re-run the NSD eval "
                f"so one layer is selected per ROI.")
        layers[region] = found.layer.iloc[0]
        rprint(f"    {region}: reusing layer {layers[region]} from NSD "
               f"({found.subject_idx.nunique()} subjects agree)", style="info")
    return layers


def _split_synthetic_stimuli(ids, seed=42):
    """Fixed stratified 50/50 split of the synthetic stimuli for in-dataset layer selection.

    Each stimulus family (e.g. ``spiral_A_sf1``, ``word4_pos2``) contributes half its
    members to selection and half to reporting, so both halves are representative.
    """
    rng = np.random.RandomState(seed)
    families = {}
    for sid in ids:
        families.setdefault(sid.rsplit("_", 1)[0], []).append(sid)
    select = set()
    for members in families.values():
        members = sorted(members)
        rng.shuffle(members)
        select.update(members[:len(members) // 2])
    return sorted(select), sorted(sid for sid in ids if sid not in select)


def _print_layer_report(selection_scores, regions, subjects):
    """Mean selection score across subjects for every layer, per region."""
    for region in regions:
        layers = [d["layer"] for d in selection_scores[region][subjects[0]]]
        means = {l: np.mean([next(d["score"] for d in selection_scores[region][s] if d["layer"] == l)
                             for s in subjects]) for l in layers}
        rprint(f"    {region}: " + "  ".join(f"{l} {m:.3f}" for l, m in means.items()), style="info")


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
    # Default to the model's full layer set, but honour an explicit
    # return_nodes (config or --override) so a run can be pinned to one layer.
    if not cfg.get("return_nodes"):
        cfg.return_nodes = list(mutils.TORCHVISION_RETURN_NODES[cfg.model_name])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = cfg.neural_dataset.lower()

    # ── THINGS-BEHAVIOR: 80/20 concept-level train/test RSA ──
    if dataset == "things-behavior":
        _print_header(cfg)
        model = mutils.load_model(cfg, dev, verbose=verbose)
        neural_data, dl = get_neural_loader(cfg)
        if "CLIP" in cfg.get("model_name", ""):
            dl.dataset.tr = _get_eval_transform(cfg)
        rprint(f"  THINGS data loaded", style="success")

        # Match prepare_concept_alignment's insertion order and missing-image filter.
        available = set(dl.dataset.keys)
        concepts = [c for c, images in neural_data["image_ids"].items()
                    if available.intersection(images)]
        perm = np.random.RandomState(42).permutation(len(concepts))
        n_sel = int(len(concepts) * 0.2)
        sel_idx, eval_idx = perm[:n_sel], perm[n_sel:]
        train_images = {sid for i in sel_idx for sid in neural_data["image_ids"][concepts[i]]}
        test_images = {sid for i in eval_idx for sid in neural_data["image_ids"][concepts[i]]}
        calibration_ids = (train_images - test_images) & available
        prepare_eval_batchnorm(model, cfg, dl, calibration_ids, dev)
        model = mutils.configure_feature_extractor(cfg, model, verbose=verbose)
        acts, ids = mutils.get_activations(model, dl, dev)

        # Merge train/test images, average activations per concept
        all_concepts = prepare_concept_alignment(cfg, acts, neural_data, ids)
        del acts, neural_data, ids
        torch.cuda.empty_cache()

        assert all_concepts.stimulus_ids == concepts, "THINGS split order changed"

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

    # ── NSD-SYNTHETIC: OOD test set, layers reused from the matching NSD run ──
    if dataset == "nsd_synthetic":
        subjects, regions = _listify(cfg.subject_idx), _listify(cfg.region)
        _print_header(cfg, len(subjects), len(regions))
        best_layers = _lookup_nsd_best_layers(cfg, subjects, regions)
        data = load_all_nsd_synthetic_data(cfg, subjects=subjects, regions=regions)
        rprint(f"  {len(data['stimuli'])} synthetic test stimuli", style="success")
        model = mutils.load_model(cfg, dev, verbose=verbose)
        bn_source = cfg.get("bn_calibration_source", "checkpoint")
        if bn_source == "dataset":
            raise ValueError("nsd_synthetic has no training split to calibrate BatchNorm on; "
                             "use bn_calibration_source=checkpoint, nsd or imagenet")
        if bn_source == "imagenet":
            prepare_eval_batchnorm(model, cfg, None, [], dev)
        elif bn_source == "nsd":
            # Same BN statistics the NSD run selected its layer with: calibrate on
            # NSD training images under the NSD transform, which hits that run's cache.
            nsd_cfg = OmegaConf.merge(cfg, {"neural_dataset": "nsd"})
            nsd = load_all_nsd_data(nsd_cfg, subjects=subjects, regions=regions)
            dl = _make_loader(nsd["stimuli"], _get_eval_transform(nsd_cfg),
                              cfg.batchsize, cfg.num_workers)
            prepare_eval_batchnorm(model, nsd_cfg, dl,
                                   training_image_ids(nsd["neural"], nsd["stimuli"].keys()), dev)
            cfg.bn_calibration = nsd_cfg.bn_calibration
            del nsd, dl
        model = mutils.configure_feature_extractor(cfg, model, verbose=verbose)
        test_ids, selection_scores = data["test_ids"], None
        if cfg.get("layer_source", "nsd") == "split":
            # Select on half the synthetic stimuli instead of inheriting from NSD.
            select_ids, test_ids = _split_synthetic_stimuli(data["test_ids"])
            rprint(f"  Split: {len(select_ids)} stimuli for selection, {len(test_ids)} for reporting",
                   style="info")
            dl = _make_loader({sid: data["stimuli"][sid] for sid in select_ids},
                              _get_eval_transform(cfg), cfg.batchsize, cfg.num_workers)
            acts, ids = mutils.get_activations(model, dl, dev)
            select_neural = {r: {s: {"train": {sid: data["neural"][r][s][sid] for sid in select_ids}}
                                 for s in subjects} for r in regions}
            method = cfg.get("compare_method", "spearman").lower()
            best_layers, selection_scores = _select_rsa_layers(
                acts, ids, select_neural, subjects, regions, method, n_select=None)
            _print_layer_report(selection_scores, regions, subjects)
            del acts, dl
        test_stimuli = {sid: data["stimuli"][sid] for sid in test_ids}
        return _reextract_and_score(model, cfg, dev, test_stimuli, test_ids,
                                    data["neural"], best_layers, regions, subjects,
                                    selection_scores, verbose)

    # ── NSD / TVSD: unified multi-subject path ──────────
    subjects = _listify(cfg.subject_idx)
    regions = _listify(cfg.region)

    _print_header(cfg, len(subjects), len(regions))

    # Load model once
    model = mutils.load_model(cfg, dev, verbose=verbose)

    # Load all neural data once
    if dataset == "nsd":
        all_data = load_all_nsd_data(cfg, subjects=subjects, regions=regions)
    elif dataset == "tvsd":
        all_data = load_all_tvsd_data(cfg, subjects=subjects, regions=regions)
        if (cfg.get("analysis", "rsa").lower() == "rsa"
                and cfg.get("tvsd_rsa_on_train", True)):
            frac = cfg.get("tvsd_select_frac", _TVSD_SELECT_FRAC)
            sel_ids, rep_ids = _tvsd_train_split(
                all_data, select_frac=frac, n_report=cfg.get("n_report", 5000))
            cfg.n_select = None  # the selection split is already the subsample
            rprint(
                f"  TVSD RSA on train stimuli: {len(sel_ids)} for layer selection "
                f"({frac:.0%}), {len(rep_ids)} for reporting "
                f"(single-trial, no noise ceiling)",
                style="info",
            )
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
    prepare_eval_batchnorm(model, cfg, dl,
                           training_image_ids(all_data["neural"], stimuli.keys()), dev)
    model = mutils.configure_feature_extractor(cfg, model, verbose=verbose)
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


# ──────────────── encoding score helper ─────────────────
def _eval_encoding(cfg, model, acts, ids, all_data, subjects, regions, verbose):
    """Per-region encoding score using SRP activations (no re-extraction).

    Phase 1: every subject scores every layer on an 80/20 split of its train
    data; the region's layer is the best on average across subjects.
    Phase 2: that layer is refit on each subject's full train data and scored
    on its test data.

    A subject's regions share the same stimuli, so their voxels are concatenated
    and fit together: one ridge fit per (subject, layer) instead of one per
    (subject, layer, region). Alpha selection and weights are per voxel, so
    the per-region scores are identical to fitting each region on its own.
    """
    neural = all_data["neural"]
    bootstrap = cfg.get("bootstrap", True)
    n_bootstrap = cfg.get("n_bootstrap", 1000)
    pca_k = cfg.get("pca_k", 1) if cfg.get("reconstruct_from_pcs") else None
    solver = cfg.get("encoding_solver", "fast")

    def _subject_data(subj):
        """Train/test AlignmentData with all regions' voxels side by side, plus {region: column slice}."""
        train, test = prepare_traintest_alignment(cfg, acts, neural[regions[0]][subj], ids)
        train_neural, test_neural, groups, start = [], [], {}, 0
        for region in regions:
            _, tr, tr_ids = _align_stimulus_level({}, neural[region][subj]["train"], ids)
            _, te, te_ids = _align_stimulus_level({}, neural[region][subj]["test"], ids)
            assert tr_ids == train.stimulus_ids and te_ids == test.stimulus_ids, \
                f"subject {subj}: {region} does not share stimuli with {regions[0]}"
            train_neural.append(tr)
            test_neural.append(te)
            groups[region] = slice(start, start + tr.size(1))
            start += tr.size(1)
        train.neural = torch.cat(train_neural, dim=1)
        test.neural = torch.cat(test_neural, dim=1)
        return train, test, groups

    rprint("\n  Phase 1: layer selection", style="info")
    selection_scores = {region: {} for region in regions}
    for subj in subjects:
        train_data, _, groups = _subject_data(subj)
        per_region = select_layer_scores(train_data, verbose=verbose, target_groups=groups, solver=solver)
        for region in regions:
            selection_scores[region][subj] = per_region[region]
        del train_data
    best_layers = {}
    for region in regions:
        best_layer, best_score = _best_layer_across_subjects(selection_scores[region])
        best_layers[region] = best_layer
        rprint(
            f"    {region}: {best_layer} (mean val r {best_score:.4f} "
            f"across {len(subjects)} subjects)",
            style="info",
        )

    rprint("  Phase 2: test evaluation", style="info")
    all_results = []
    region_scores = {region: [] for region in regions}
    for subj in subjects:
        train_data, test_data, groups = _subject_data(subj)
        for layer in sorted(set(best_layers.values())):
            layer_groups = {r: groups[r] for r in regions if best_layers[r] == layer}
            results = evaluate_layer(
                layer, train_data, test_data,
                bootstrap=bootstrap, n_bootstrap=n_bootstrap,
                verbose=verbose, reconstruct_pca_k=pca_k, target_groups=layer_groups, solver=solver,
            )
            for region, result in results.items():
                result["layer_selection_scores"] = selection_scores[region][subj]
                result["region"] = region
                result["subject_idx"] = subj

                if cfg.get("log_expdata"):
                    save_cfg = OmegaConf.merge(cfg, {"subject_idx": subj, "region": region})
                    save_results(pd.DataFrame([result]), save_cfg, quiet=True)

                all_results.append(result)
                region_scores[region].append(result["score"])
        del train_data, test_data

    region_means = {}
    for region in regions:
        _print_region_results(region, best_layers[region], region_scores[region], subjects)
        if cfg.get("log_expdata"):
            rprint(f"    Saved {len(subjects)} results to results.db", style="success")
        region_means[region] = np.mean(region_scores[region])

    _print_cross_region_summary(region_means)

    # Free bulk activations and model
    del acts, model
    torch.cuda.empty_cache()

    return pd.DataFrame(all_results)
