"""Alignment math on synthetic data: RSA, encoding score, layer selection, DB storage.

Every test runs on CPU in seconds. The synthetic model has two layers: ``good``
generates the neural data through a linear map, ``noise`` is unrelated. A
correct pipeline must select ``good`` and score it highly.
"""
import json
import sqlite3

import numpy as np
import pytest
import rsatoolbox
import scipy.stats
import torch
from omegaconf import OmegaConf

from visreps.analysis.alignment import AlignmentData, _align_stimulus_level, prepare_concept_alignment
from visreps.analysis.encoding_score import (
    _fit_and_score, compute_encoding_score, evaluate_layer, select_layer_scores,
)
from visreps.analysis.rsa import (
    _kendall_tau_a, compute_rdm, compute_rdm_correlation, compute_rsa, score_rdm_pair,
)
from visreps.evals import _best_layer_across_subjects
import visreps.utils as vu


# ───────────────────────── synthetic data ─────────────────────────
def _make_split(n, rng, W, noise=0.1):
    good = rng.randn(n, 30).astype(np.float32)
    neural = good @ W + noise * rng.randn(n, W.shape[1]).astype(np.float32)
    acts = {"good": torch.from_numpy(good), "noise": torch.from_numpy(rng.randn(n, 30).astype(np.float32))}
    return AlignmentData(acts, torch.from_numpy(neural), stimulus_ids=[str(i) for i in range(n)])


@pytest.fixture
def synthetic():
    rng = np.random.RandomState(0)
    W = rng.randn(30, 20).astype(np.float32)
    return _make_split(200, rng, W), _make_split(60, rng, W)


# ───────────────────────── RSA primitives ─────────────────────────
def test_rdm_matches_rsatoolbox():
    x = np.random.RandomState(1).randn(30, 50).astype(np.float32)
    ref = rsatoolbox.rdm.calc_rdm(rsatoolbox.data.Dataset(x), method="correlation").get_matrices()[0]
    np.testing.assert_allclose(compute_rdm(torch.from_numpy(x)).numpy(), ref, atol=1e-4)


@pytest.mark.parametrize("ours,ref,atol", [("Spearman", "spearman", 1e-4), ("Kendall", "tau-a", 1e-3)])
def test_rdm_comparison_matches_rsatoolbox(ours, ref, atol):
    x = np.random.RandomState(1).randn(25, 40).astype(np.float32)
    y = np.random.RandomState(2).randn(25, 40).astype(np.float32)
    score = compute_rdm_correlation(compute_rdm(torch.from_numpy(x)), compute_rdm(torch.from_numpy(y)), correlation=ours)
    ds = [rsatoolbox.rdm.calc_rdm(rsatoolbox.data.Dataset(a), method="correlation") for a in (x, y)]
    assert score == pytest.approx(rsatoolbox.rdm.compare(ds[0], ds[1], method=ref)[0, 0], abs=atol)


def test_kendall_tau_a_matches_brute_force():
    rng = np.random.RandomState(3)
    a, b = rng.randn(150), rng.randn(150)
    i, j = np.triu_indices(150, 1)
    expected = np.mean(np.sign(a[i] - a[j]) * np.sign(b[i] - b[j]))
    assert _kendall_tau_a(a, b)[0] == pytest.approx(expected, abs=1e-12)


def test_bootstrap_is_deterministic_and_brackets_score():
    x = torch.randn(80, 40, generator=torch.Generator().manual_seed(0))
    y = x + 0.5 * torch.randn(80, 40, generator=torch.Generator().manual_seed(1))
    r1 = score_rdm_pair(compute_rdm(x), compute_rdm(y), "spearman", bootstrap=True, n_bootstrap=50)
    r2 = score_rdm_pair(compute_rdm(x), compute_rdm(y), "spearman", bootstrap=True, n_bootstrap=50)
    score, lo, hi, boots = r1
    assert lo <= score <= hi and len(boots) == 50
    assert r1 == r2


# ───────────────────────── RSA pipeline ─────────────────────────
def test_rsa_selects_generating_layer_and_uses_exact_reextraction(synthetic):
    sel, ev = synthetic
    cfg = {"compare_method": "spearman"}
    calls = []

    def re_extract(layer, sids):
        calls.append(layer)
        return ev.activations[layer], sids

    (res,) = compute_rsa(cfg, sel, ev, bootstrap=False, re_extract_fn=re_extract, quiet=True)
    assert res["layer"] == "good" and calls == ["good"]
    assert res["score"] > 0.5
    assert {s["layer"] for s in res["layer_selection_scores"]} == {"good", "noise"}


def test_rsa_layer_selection_ignores_test_data(synthetic):
    sel, ev = synthetic
    cfg = {"compare_method": "spearman"}
    a = compute_rsa(cfg, sel, ev, bootstrap=False, quiet=True)[0]["layer_selection_scores"]
    ev_shuffled = AlignmentData(ev.activations, ev.neural[torch.randperm(60)], stimulus_ids=ev.stimulus_ids)
    b = compute_rsa(cfg, sel, ev_shuffled, bootstrap=False, quiet=True)[0]["layer_selection_scores"]
    assert a == b


# ───────────────────────── encoding score ─────────────────────────
def test_ridge_recovers_exact_linear_map():
    from himalaya.backend import set_backend
    backend = set_backend("torch_cuda", on_error="warn")
    rng = np.random.RandomState(0)
    X, W = rng.randn(300, 20).astype(np.float32), rng.randn(20, 5).astype(np.float32)
    Y = X @ W
    _, r = _fit_and_score(backend.asarray(X[:200]), backend.asarray(Y[:200]),
                          torch.from_numpy(X[200:]), backend.asarray(Y[200:]), np.logspace(-6, 2, 9), backend)
    assert r > 0.99


def test_encoding_selects_generating_layer_and_scores_it(synthetic):
    sel, ev = synthetic
    scores = select_layer_scores(sel)
    assert max(scores, key=lambda s: s["score"])["layer"] == "good"
    res = evaluate_layer("good", sel, ev, bootstrap=True, n_bootstrap=30)
    assert res["score"] > 0.9 and res["ci_low"] <= res["score"] <= res["ci_high"]
    assert res["compare_method"] == "pearson" and len(res["bootstrap_scores"]) == 30
    (wrapped,) = compute_encoding_score(sel, ev, bootstrap=False, quiet=True)
    assert wrapped["layer"] == "good"


def test_encoding_layer_selection_ignores_test_data(synthetic):
    sel, ev = synthetic
    assert select_layer_scores(sel) == select_layer_scores(
        AlignmentData(sel.activations, sel.neural, stimulus_ids=sel.stimulus_ids))


# ───────────────────────── one layer per ROI ─────────────────────────
def test_best_layer_across_subjects_uses_mean_not_any_single_subject():
    scores = {
        0: [{"layer": "A", "score": 0.50}, {"layer": "B", "score": 0.49}],
        1: [{"layer": "A", "score": 0.20}, {"layer": "B", "score": 0.40}],
        2: [{"layer": "A", "score": 0.20}, {"layer": "B", "score": 0.40}],
    }
    layer, mean = _best_layer_across_subjects(scores)
    assert layer == "B" and mean == pytest.approx(0.43)


# ───────────────────────── stimulus / concept alignment ─────────────────────────
def test_align_stimulus_level_matches_ids_and_order():
    acts = {"l": torch.arange(5).float()[:, None]}
    targets = {"3": np.array([30.0]), "1": np.array([10.0])}
    a, neural, ids = _align_stimulus_level(acts, targets, ["0", "1", "2", "3", "4"])
    assert ids == ["1", "3"]
    assert a["l"].squeeze().tolist() == [1.0, 3.0] and neural.squeeze().tolist() == [10.0, 30.0]


def test_prepare_concept_alignment_averages_images_per_concept():
    acts = {"l": torch.tensor([[1.0], [3.0], [10.0]])}
    neural = {"embeddings": {"cat": np.zeros(2, np.float32), "dog": np.ones(2, np.float32)},
              "image_ids": {"cat": ["c1", "c2"], "dog": ["d1"], "missing": ["m1"]}}
    data = prepare_concept_alignment({}, acts, neural, ["c1", "c2", "d1"])
    assert data.stimulus_ids == ["cat", "dog"]
    assert data.activations["l"].squeeze().tolist() == [2.0, 10.0]
    assert data.concept_image_ids == {"cat": ["c1", "c2"], "dog": ["d1"]}


# ───────────────────────── results.db ─────────────────────────
def test_save_results_round_trip_and_dedup():
    cfg = OmegaConf.create({
        "seed": 1, "epoch": 20, "region": "IT", "subject_idx": 0, "neural_dataset": "tvsd",
        "cfg_id": 8, "pca_labels": True, "pca_n_classes": 8, "pca_labels_folder": "pca_labels_alexnet",
        "checkpoint_dir": "/x", "analysis": "rsa", "compare_method": "spearman",
        "reconstruct_from_pcs": False, "pca_k": 1, "model_name": "CustomCNN",
    })
    row = {"layer": "conv3_pre", "compare_method": "spearman", "score": 0.3, "ci_low": 0.2, "ci_high": 0.4,
           "analysis": "rsa", "layer_selection_scores": [{"layer": "conv3_pre", "score": 0.25}],
           "bootstrap_scores": [0.29, 0.31]}
    import pandas as pd
    vu.save_results(pd.DataFrame([row]), cfg, quiet=True)
    vu.save_results(pd.DataFrame([dict(row, layer="fc1_post")]), cfg, quiet=True)  # rerun replaces

    conn = sqlite3.connect(vu._RESULTS_DB_PATH)
    results = pd.read_sql("SELECT * FROM results", conn)
    assert len(results) == 1 and results.iloc[0]["layer"] == "fc1_post"
    assert results.iloc[0]["run_id"] == vu._compute_run_id(cfg)
    assert json.loads(pd.read_sql("SELECT scores FROM bootstrap_distributions", conn).iloc[0, 0]) == [0.29, 0.31]
    assert len(pd.read_sql("SELECT * FROM layer_selection_scores", conn)) == 1
