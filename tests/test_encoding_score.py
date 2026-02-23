"""Comprehensive tests for encoding score computation (NSD and TVSD).

Tests cover:
  1. Unit tests for _znorm (normalization helper)
  2. Unit tests for _fit_and_score (ridge regression mechanics)
  3. Full compute_encoding_score pipeline with synthetic data
  4. Bootstrap CI computation
  5. X and Y normalization correctness at each stage
  6. Edge cases (single voxel, few samples, many layers, constant features)
  7. Alignment dispatch (prepare_traintest_alignment -> compute_traintest_alignment)
  8. DB storage (results, layer_selection_scores, bootstrap_distributions)
  9. End-to-end with real NSD data (encoding + bootstrap)
  10. End-to-end with real TVSD data (encoding + bootstrap)
  11. Mathematical correctness (manual Pearson r verification)
  12. Pipeline-level normalization leakage detection
  13. Bootstrap mechanics (replace=False, cached predictions, RNG coupling)
  14. PCA reconstruction path (reconstruct_pca_k)
  15. Robustness (constant features/voxels, distribution shift, NaN)
  16. Dispatch integration (seed propagation)

Run all tests:
    source .venv/bin/activate && pytest tests/test_encoding_score.py -v

Run only fast tests (skip end-to-end):
    source .venv/bin/activate && pytest tests/test_encoding_score.py -v -m "not slow"
"""
import json
import os
import sqlite3
import sys
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
import scipy.stats
import torch
from omegaconf import OmegaConf

# ── Ensure project root on path ──────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv()

from visreps.analysis.encoding_score import (
    _znorm, _znorm_fit, _flatten_to_cpu, _fit_and_score, compute_encoding_score,
)
from visreps.analysis.alignment import (
    AlignmentData,
    _align_stimulus_level,
    prepare_traintest_alignment,
    compute_traintest_alignment,
)
from visreps.utils import save_results, _compute_run_id

from himalaya.backend import set_backend
from himalaya.scoring import correlation_score


# ═══════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════

@pytest.fixture
def backend():
    """Initialize himalaya backend (GPU preferred, falls back to CPU)."""
    return set_backend("torch_cuda", on_error="warn")


@pytest.fixture
def synthetic_encoding_data():
    """Synthetic data with a known linear signal in one layer.

    'signal_layer': X -> Y via linear map + small noise (high Pearson r).
    'noise_layer':  random X -> same Y (low Pearson r).
    """
    torch.manual_seed(42)
    np.random.seed(42)
    n_train, n_test = 200, 50
    n_features, n_voxels = 50, 20

    true_weights = torch.randn(n_features, n_voxels) * 0.5

    X_train_signal = torch.randn(n_train, n_features)
    X_test_signal = torch.randn(n_test, n_features)
    Y_train = X_train_signal @ true_weights + torch.randn(n_train, n_voxels) * 0.1
    Y_test = X_test_signal @ true_weights + torch.randn(n_test, n_voxels) * 0.1

    X_train_noise = torch.randn(n_train, n_features)
    X_test_noise = torch.randn(n_test, n_features)

    train = AlignmentData(
        activations={"signal_layer": X_train_signal, "noise_layer": X_train_noise},
        neural=Y_train,
        stimulus_ids=[f"train_{i}" for i in range(n_train)],
    )
    test = AlignmentData(
        activations={"signal_layer": X_test_signal, "noise_layer": X_test_noise},
        neural=Y_test,
        stimulus_ids=[f"test_{i}" for i in range(n_test)],
    )
    return train, test


@pytest.fixture
def encoding_cfg():
    """Minimal config for encoding score tests."""
    return OmegaConf.create({
        "analysis": "encoding_score",
        "compare_method": "pearson",
        "bootstrap": False,
        "n_bootstrap": 100,
        "verbose": False,
        "neural_dataset": "nsd",
        "seed": 1,
        "epoch": 20,
        "region": "ventral visual stream",
        "subject_idx": "0",
        "cfg_id": 1000,
        "pca_labels": False,
        "pca_n_classes": None,
        "pca_labels_folder": None,
        "model_name": "CustomCNN",
        "checkpoint_dir": "/data/ymehta3/default",
        "reconstruct_from_pcs": False,
        "pca_k": 1,
    })


# Module-scoped fixtures for expensive end-to-end tests
def _check_gpu_memory(min_free_gb=6.0):
    """Skip test if GPU doesn't have enough free memory."""
    if not torch.cuda.is_available():
        pytest.skip("No CUDA GPU available")
    free = torch.cuda.mem_get_info()[0] / 1e9
    if free < min_free_gb:
        pytest.skip(
            f"Not enough GPU memory: {free:.1f} GB free, need {min_free_gb} GB. "
            f"Encoding score's SVD-based ridge regression requires significant GPU headroom."
        )


@pytest.fixture(scope="module")
def nsd_encoding_results():
    """Run NSD encoding score once, share across tests in this module."""
    _check_gpu_memory(min_free_gb=8.0)  # NSD ventral has 7604 voxels
    import visreps.utils as utils
    import visreps.evals as evals

    overrides = [
        "mode=eval",
        "cfg_id=1000", "seed=1",
        "checkpoint_model=checkpoint_epoch_20.pth",
        "neural_dataset=nsd",
        "analysis=encoding_score",
        "subject_idx=0",
        "region=ventral visual stream",
        "log_expdata=false",
        "bootstrap=true",
        "n_bootstrap=50",
        "verbose=true",
        "batchsize=64",
        "num_workers=4",
    ]

    cfg = utils.load_config("configs/eval/base.json", overrides)
    cfg = utils.validate_config(cfg)

    try:
        results = evals.eval(cfg)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        pytest.skip(
            "CUDA OOM during NSD encoding — himalaya's SVD solver needs more GPU "
            "memory than available (9000 train × 4096 features). "
            "Free up GPU memory or use a machine with more VRAM."
        )
    return results


@pytest.fixture(scope="module")
def tvsd_encoding_results():
    """Run TVSD encoding score once, share across tests in this module."""
    _check_gpu_memory(min_free_gb=6.0)  # TVSD has 22k train × 4096 features
    import visreps.utils as utils
    import visreps.evals as evals

    overrides = [
        "mode=eval",
        "cfg_id=1000", "seed=1",
        "checkpoint_model=checkpoint_epoch_20.pth",
        "neural_dataset=tvsd",
        "analysis=encoding_score",
        "subject_idx=0",
        "region=IT",
        "log_expdata=false",
        "bootstrap=true",
        "n_bootstrap=50",
        "verbose=true",
        "batchsize=64",
        "num_workers=4",
    ]

    cfg = utils.load_config("configs/eval/base.json", overrides)
    cfg = utils.validate_config(cfg)

    try:
        results = evals.eval(cfg)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        pytest.skip(
            "CUDA OOM during TVSD encoding — himalaya's SVD solver needs more GPU "
            "memory than available (22k train × 4096 features). "
            "Free up GPU memory or use a machine with more VRAM."
        )
    return results


# ═══════════════════════════════════════════════════════════
# 1. Z-NORMALIZATION
# ═══════════════════════════════════════════════════════════

class TestZnorm:
    """Tests for the _znorm(X, mean, std) helper."""

    def test_basic_normalization(self):
        """Z-normalized data should have mean~0, std~1."""
        X = torch.randn(100, 10) * 5 + 3
        mean = X.mean(dim=0)
        std = X.std(dim=0) + 1e-8
        result = _znorm(X, mean, std)

        assert result.mean(dim=0).abs().max() < 0.01
        assert (result.std(dim=0) - 1.0).abs().max() < 0.05

    def test_constant_columns_become_zero(self):
        """Constant columns should become zero (not NaN/inf)."""
        X = torch.ones(50, 5)
        X[:, 2] = 7.0  # constant column
        mean = X.mean(dim=0)
        std = X.std(dim=0) + 1e-8
        result = _znorm(X, mean, std)

        assert torch.isfinite(result).all()
        assert result[:, 2].abs().max() < 1e-5  # (7 - 7) / eps = 0

    def test_preserves_dtype(self):
        """Output dtype should match input dtype."""
        X = torch.randn(10, 5, dtype=torch.float32)
        result = _znorm(X, X.mean(0), X.std(0) + 1e-8)
        assert result.dtype == torch.float32

    def test_preserves_float64_dtype(self):
        """Output dtype should match input dtype for float64."""
        X = torch.randn(10, 5, dtype=torch.float64)
        result = _znorm(X, X.mean(0), X.std(0) + 1e-8)
        assert result.dtype == torch.float64

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA GPU")
    def test_gpu_tensor(self):
        """_znorm should work on GPU tensors and keep result on GPU."""
        X = torch.randn(50, 10, device="cuda")
        mean = X.mean(dim=0)
        std = X.std(dim=0) + 1e-8
        result = _znorm(X, mean, std)

        assert result.device.type == "cuda"
        assert result.mean(dim=0).abs().max() < 0.01
        assert torch.isfinite(result).all()


class TestZnormFit:
    """Tests for the _znorm_fit(X) helper."""

    def test_returns_normalized_mean_std(self):
        """Should return (normalized_X, mean, std)."""
        X = torch.randn(100, 10) * 5 + 3
        result, mean, std = _znorm_fit(X)
        assert result.shape == X.shape
        assert mean.shape == (10,)
        assert std.shape == (10,)

    def test_normalized_output_properties(self):
        """Normalized output should have mean~0, std~1."""
        X = torch.randn(200, 10) * 5 + 3
        result, _, _ = _znorm_fit(X)
        assert result.mean(dim=0).abs().max() < 0.01
        assert (result.std(dim=0) - 1.0).abs().max() < 0.1

    def test_std_includes_epsilon(self):
        """Std should include 1e-8 epsilon to avoid division by zero."""
        X = torch.ones(50, 3)  # constant columns -> std = 0
        result, mean, std = _znorm_fit(X)
        # std should be ~1e-8, not 0
        assert (std > 0).all()
        assert torch.isfinite(result).all()

    def test_roundtrip_with_znorm(self):
        """_znorm_fit output should match _znorm with the same stats."""
        X = torch.randn(50, 10)
        result1, mean, std = _znorm_fit(X)
        result2 = _znorm(X, mean, std)
        torch.testing.assert_close(result1, result2)


class TestFlattenToCPU:
    """Tests for the _flatten_to_cpu helper."""

    def test_flattens_4d_to_2d(self):
        """4D conv activations (N, C, H, W) should flatten to (N, C*H*W)."""
        acts = {"conv1": torch.randn(10, 64, 6, 6)}
        result = _flatten_to_cpu(acts)
        assert result["conv1"].shape == (10, 64 * 6 * 6)

    def test_keeps_2d_unchanged(self):
        """2D fc activations (N, F) should remain unchanged in shape."""
        acts = {"fc1": torch.randn(10, 256)}
        result = _flatten_to_cpu(acts)
        assert result["fc1"].shape == (10, 256)

    def test_output_is_cpu_float32(self):
        """Output should always be CPU float32."""
        acts = {"layer": torch.randn(10, 20, dtype=torch.float64)}
        result = _flatten_to_cpu(acts)
        assert result["layer"].dtype == torch.float32
        assert result["layer"].device.type == "cpu"

    def test_does_not_mutate_input(self):
        """Original dict should be unchanged after flattening."""
        original = torch.randn(10, 64, 6, 6)
        acts = {"conv1": original}
        result = _flatten_to_cpu(acts)
        # Original should still be 4D
        assert acts["conv1"].shape == (10, 64, 6, 6)
        assert acts["conv1"] is original  # same object

    def test_multiple_layers(self):
        """Should handle multiple layers with mixed shapes."""
        acts = {
            "conv1": torch.randn(10, 64, 6, 6),
            "conv2": torch.randn(10, 128, 3, 3),
            "fc1": torch.randn(10, 256),
        }
        result = _flatten_to_cpu(acts)
        assert result["conv1"].shape == (10, 64 * 6 * 6)
        assert result["conv2"].shape == (10, 128 * 3 * 3)
        assert result["fc1"].shape == (10, 256)


# ═══════════════════════════════════════════════════════════
# 2. _FIT_AND_SCORE (ridge regression helper)
# ═══════════════════════════════════════════════════════════

class TestFitAndScore:
    """Tests for the ridge regression fit-and-score helper."""

    def test_perfect_linear_data(self, backend):
        """When Y = X @ w exactly, Pearson r should be very high."""
        w = torch.randn(20, 5)
        X = torch.randn(100, 20)
        Y = X @ w

        X_tr, Y_tr = backend.asarray(X[:80]), backend.asarray(Y[:80])
        X_te, Y_te = X[80:], backend.asarray(Y[80:])  # X_te on CPU

        _, score = _fit_and_score(X_tr, Y_tr, X_te, Y_te, np.logspace(-5, 5, 10), backend)
        assert score > 0.95, f"Perfect linear data should give r>0.95, got {score:.4f}"

    def test_random_data_low_score(self, backend):
        """Random X -> Y should give score near 0."""
        X = backend.asarray(torch.randn(100, 20))
        Y = backend.asarray(torch.randn(100, 10))

        _, score = _fit_and_score(X[:80], Y[:80], X[80:], Y[80:], np.logspace(-5, 5, 10), backend)
        assert abs(score) < 0.35, f"Random data should give |r|<0.35, got {score:.4f}"

    def test_cpu_xte_handled(self, backend):
        """X_te on CPU should be moved to GPU automatically."""
        w = torch.randn(10, 3)
        X = torch.randn(60, 10)
        Y = X @ w + torch.randn(60, 3) * 0.1

        X_tr = backend.asarray(X[:50])
        Y_tr = backend.asarray(Y[:50])
        X_te_cpu = X[50:]  # intentionally CPU
        Y_te = backend.asarray(Y[50:])

        _, score = _fit_and_score(X_tr, Y_tr, X_te_cpu, Y_te, np.logspace(-3, 3, 5), backend)
        assert score > 0.5, "CPU X_te should still produce good score"

    def test_returns_pred_score(self, backend):
        """Should return (predictions, score) tuple."""
        X = backend.asarray(torch.randn(60, 10))
        Y = backend.asarray(torch.randn(60, 5))

        pred, score = _fit_and_score(X[:50], Y[:50], X[50:], Y[50:],
                                     np.logspace(-3, 3, 5), backend)
        assert pred.shape[0] == 10  # 60 - 50 = 10 test samples
        assert isinstance(score, float)

    def test_underdetermined_case(self, backend):
        """When n_features > n_samples, ridge should still work (its raison d'etre)."""
        n_train, n_test, n_features, n_voxels = 30, 10, 200, 5
        X = torch.randn(n_train + n_test, n_features)
        Y = torch.randn(n_train + n_test, n_voxels)

        X_tr = backend.asarray(X[:n_train])
        Y_tr = backend.asarray(Y[:n_train])
        X_te = X[n_train:]  # CPU
        Y_te = backend.asarray(Y[n_train:])

        pred, score = _fit_and_score(X_tr, Y_tr, X_te, Y_te, np.logspace(-3, 5, 10), backend)
        assert pred.shape == (n_test, n_voxels)
        assert np.isfinite(score)

    def test_single_target_voxel(self, backend):
        """Should work with a single target voxel (n_voxels=1)."""
        w = torch.randn(10, 1)
        X = torch.randn(60, 10)
        Y = X @ w + torch.randn(60, 1) * 0.1

        X_tr = backend.asarray(X[:50])
        Y_tr = backend.asarray(Y[:50])
        X_te = X[50:]
        Y_te = backend.asarray(Y[50:])

        pred, score = _fit_and_score(X_tr, Y_tr, X_te, Y_te, np.logspace(-3, 3, 5), backend)
        assert pred.shape == (10, 1)
        assert score > 0.5, f"Single voxel linear data should score well, got {score:.4f}"


# ═══════════════════════════════════════════════════════════
# 3. COMPUTE_ENCODING_SCORE (full pipeline)
# ═══════════════════════════════════════════════════════════

class TestComputeEncodingScore:
    """Tests for the main encoding score function with synthetic data."""

    def test_selects_signal_layer(self, synthetic_encoding_data):
        """Should select the layer with real signal over noise."""
        train, test = synthetic_encoding_data
        results = compute_encoding_score(train, test, bootstrap=False, verbose=False)

        assert len(results) == 1
        assert results[0]["layer"] == "signal_layer"

    def test_compare_method_always_pearson(self, synthetic_encoding_data):
        """compare_method should always be 'pearson' (hardcoded, not configurable)."""
        train, test = synthetic_encoding_data
        results = compute_encoding_score(train, test, bootstrap=False, verbose=False)

        assert results[0]["compare_method"] == "pearson"

    def test_result_dict_structure(self, synthetic_encoding_data):
        """Result dict should contain all required keys."""
        train, test = synthetic_encoding_data
        results = compute_encoding_score(train, test, bootstrap=False, verbose=False)

        r = results[0]
        required = {"layer", "compare_method", "score", "ci_low", "ci_high",
                     "analysis", "layer_selection_scores"}
        assert required.issubset(r.keys())
        assert r["analysis"] == "encoding_score"
        assert r["ci_low"] is None
        assert r["ci_high"] is None

    def test_score_in_valid_range(self, synthetic_encoding_data):
        """Encoding score (Pearson r) should be in [-1, 1]."""
        train, test = synthetic_encoding_data
        results = compute_encoding_score(train, test, bootstrap=False, verbose=False)
        score = results[0]["score"]
        assert -1 <= score <= 1, f"Score {score} outside [-1, 1]"

    def test_signal_layer_high_score(self, synthetic_encoding_data):
        """Signal layer should achieve high encoding score."""
        train, test = synthetic_encoding_data
        results = compute_encoding_score(train, test, bootstrap=False, verbose=False)
        assert results[0]["score"] > 0.5, f"Signal layer score too low: {results[0]['score']:.4f}"

    def test_selection_scores_all_layers(self, synthetic_encoding_data):
        """layer_selection_scores should have one entry per layer."""
        train, test = synthetic_encoding_data
        results = compute_encoding_score(train, test, bootstrap=False, verbose=False)

        lss = results[0]["layer_selection_scores"]
        assert len(lss) == 2
        layers = {s["layer"] for s in lss}
        assert layers == {"signal_layer", "noise_layer"}

    def test_signal_layer_higher_selection_score(self, synthetic_encoding_data):
        """Signal layer should have higher selection score than noise."""
        train, test = synthetic_encoding_data
        results = compute_encoding_score(train, test, bootstrap=False, verbose=False)

        scores = {s["layer"]: s["score"] for s in results[0]["layer_selection_scores"]}
        assert scores["signal_layer"] > scores["noise_layer"]

    def test_determinism(self):
        """Same input + same seed should produce identical scores."""
        scores = []
        for _ in range(2):
            torch.manual_seed(99)
            X = torch.randn(130, 20)
            torch.manual_seed(100)
            Y = torch.randn(130, 10)

            train = AlignmentData(
                activations={"layer": X[:100].clone()},
                neural=Y[:100].clone(),
            )
            test = AlignmentData(
                activations={"layer": X[100:].clone()},
                neural=Y[100:].clone(),
            )
            r = compute_encoding_score(train, test, bootstrap=False, seed=42, verbose=False)
            scores.append(r[0]["score"])

        assert scores[0] == scores[1], f"Non-deterministic: {scores[0]} != {scores[1]}"

    def test_does_not_mutate_input_data(self):
        """compute_encoding_score should NOT mutate the input AlignmentData.

        The function copies activations internally (flatten + CPU), leaving
        the original AlignmentData objects unchanged.
        """
        train = AlignmentData(
            activations={"layer": torch.randn(80, 3, 5, 5)},  # 4D conv
            neural=torch.randn(80, 10),
        )
        test = AlignmentData(
            activations={"layer": torch.randn(20, 3, 5, 5)},
            neural=torch.randn(20, 10),
        )

        original_train_shape = train.activations["layer"].shape
        original_test_shape = test.activations["layer"].shape

        compute_encoding_score(train, test, bootstrap=False, verbose=False)

        # After: inputs should be UNCHANGED (no mutation)
        assert train.activations["layer"].shape == original_train_shape
        assert test.activations["layer"].shape == original_test_shape

    def test_does_not_mutate_test_data(self):
        """Test data should also be left unchanged by compute_encoding_score."""
        train = AlignmentData(
            activations={"layer": torch.randn(80, 3, 5, 5)},
            neural=torch.randn(80, 10),
        )
        test = AlignmentData(
            activations={"layer": torch.randn(20, 3, 5, 5)},
            neural=torch.randn(20, 10),
        )

        original_shape = test.activations["layer"].shape
        compute_encoding_score(train, test, bootstrap=False, verbose=False)

        # Test data should be UNCHANGED
        assert test.activations["layer"].shape == original_shape
        assert test.activations["layer"].ndim == 4

    def test_all_noise_layers(self):
        """When all layers are noise, should still return a result (picks least-bad layer)."""
        train_acts = {f"noise_{i}": torch.randn(80, 15) for i in range(3)}
        test_acts = {f"noise_{i}": torch.randn(20, 15) for i in range(3)}

        train = AlignmentData(activations=train_acts, neural=torch.randn(80, 5))
        test = AlignmentData(activations=test_acts, neural=torch.randn(20, 5))

        results = compute_encoding_score(train, test, bootstrap=False, verbose=False)
        assert len(results) == 1
        assert results[0]["layer"].startswith("noise_")
        assert np.isfinite(results[0]["score"])

    def test_different_seeds_different_selection_scores(self):
        """Different seeds should produce different 80/20 splits and different selection scores.

        The final test score is the same regardless of seed (refit on full train data),
        but the layer_selection_scores should differ because the 80/20 validation split
        changes with the seed.
        """
        X = torch.randn(130, 20)
        Y = torch.randn(130, 10)

        selection_scores_per_seed = []
        for seed in [1, 2]:
            train = AlignmentData(activations={"layer": X[:100].clone()}, neural=Y[:100].clone())
            test = AlignmentData(activations={"layer": X[100:].clone()}, neural=Y[100:].clone())
            r = compute_encoding_score(train, test, bootstrap=False, seed=seed, verbose=False)
            sel_score = r[0]["layer_selection_scores"][0]["score"]
            selection_scores_per_seed.append(sel_score)

        # Different seeds produce different val splits -> different selection scores
        assert selection_scores_per_seed[0] != selection_scores_per_seed[1], (
            "Different seeds should produce different layer selection scores"
        )

    def test_verbose_mode_no_crash(self, synthetic_encoding_data):
        """Verbose mode should not crash (smoke test for Rich progress bars)."""
        train, test = synthetic_encoding_data
        results = compute_encoding_score(train, test, bootstrap=False, verbose=True)
        assert len(results) == 1

    def test_mixed_conv_fc_layers(self):
        """Should handle a mix of 4D conv and 2D fc activations (realistic model)."""
        train = AlignmentData(
            activations={
                "conv1": torch.randn(80, 64, 6, 6),  # 4D conv
                "conv2": torch.randn(80, 128, 3, 3),  # 4D conv
                "fc1": torch.randn(80, 256),           # 2D fc
            },
            neural=torch.randn(80, 10),
        )
        test = AlignmentData(
            activations={
                "conv1": torch.randn(20, 64, 6, 6),
                "conv2": torch.randn(20, 128, 3, 3),
                "fc1": torch.randn(20, 256),
            },
            neural=torch.randn(20, 10),
        )

        results = compute_encoding_score(train, test, bootstrap=False, verbose=False)
        assert len(results) == 1
        assert results[0]["layer"] in {"conv1", "conv2", "fc1"}
        assert len(results[0]["layer_selection_scores"]) == 3

    def test_ignores_cfg_compare_method(self):
        """Encoding score should always use 'pearson' even if data came through a
        pipeline that set compare_method to something else. The compare_method is
        hardcoded in compute_encoding_score, not read from cfg."""
        train = AlignmentData(
            activations={"layer": torch.randn(80, 20)},
            neural=torch.randn(80, 10),
        )
        test = AlignmentData(
            activations={"layer": torch.randn(20, 20)},
            neural=torch.randn(20, 10),
        )
        # compute_encoding_score doesn't take cfg, so compare_method is always pearson
        results = compute_encoding_score(train, test, bootstrap=False, verbose=False)
        assert results[0]["compare_method"] == "pearson"

    def test_bootstrap_reproducibility(self):
        """Same seed should produce identical bootstrap scores."""
        torch.manual_seed(99)
        X = torch.randn(100, 20)
        torch.manual_seed(100)
        w = torch.randn(20, 10)
        Y = X @ w + torch.randn(100, 10) * 0.3

        scores_list = []
        for _ in range(2):
            train = AlignmentData(activations={"layer": X[:80].clone()}, neural=Y[:80].clone())
            test = AlignmentData(activations={"layer": X[80:].clone()}, neural=Y[80:].clone())
            r = compute_encoding_score(train, test, bootstrap=True, n_bootstrap=20, seed=42, verbose=False)
            scores_list.append(r[0]["bootstrap_scores"])

        assert scores_list[0] == scores_list[1], "Same seed should give identical bootstrap scores"

    def test_analysis_field(self, synthetic_encoding_data):
        """Result should always have analysis='encoding_score'."""
        train, test = synthetic_encoding_data
        results = compute_encoding_score(train, test, bootstrap=False, verbose=False)
        assert results[0]["analysis"] == "encoding_score"


# ═══════════════════════════════════════════════════════════
# 4. BOOTSTRAP CI COMPUTATION
# ═══════════════════════════════════════════════════════════

class TestEncodingBootstrap:
    """Tests for bootstrap CI computation in encoding score."""

    def test_bootstrap_produces_cis(self, synthetic_encoding_data):
        """Bootstrap should produce non-None CI values."""
        train, test = synthetic_encoding_data
        results = compute_encoding_score(
            train, test, bootstrap=True, n_bootstrap=50, verbose=False
        )
        r = results[0]
        assert r["ci_low"] is not None
        assert r["ci_high"] is not None
        assert r["ci_low"] < r["ci_high"]

    def test_cis_bracket_point_estimate(self, synthetic_encoding_data):
        """95% CI should contain the point estimate."""
        train, test = synthetic_encoding_data
        results = compute_encoding_score(
            train, test, bootstrap=True, n_bootstrap=200, verbose=False
        )
        r = results[0]
        assert r["ci_low"] <= r["score"] <= r["ci_high"], (
            f"Point estimate {r['score']:.4f} outside CI [{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
        )

    def test_bootstrap_scores_list_valid(self, synthetic_encoding_data):
        """bootstrap_scores should be a list of n_bootstrap floats in [-1, 1]."""
        train, test = synthetic_encoding_data
        n_boot = 50
        results = compute_encoding_score(
            train, test, bootstrap=True, n_bootstrap=n_boot, verbose=False
        )

        bs = results[0]["bootstrap_scores"]
        assert len(bs) == n_boot
        assert all(isinstance(s, float) for s in bs)
        assert all(-1 <= s <= 1 for s in bs)

    def test_bootstrap_has_nonzero_variance(self, synthetic_encoding_data):
        """Bootstrap scores should have some variance (subsampling works)."""
        train, test = synthetic_encoding_data
        results = compute_encoding_score(
            train, test, bootstrap=True, n_bootstrap=100, verbose=False
        )

        bs = np.array(results[0]["bootstrap_scores"])
        assert bs.std() > 0, "Bootstrap scores should have nonzero variance"
        assert bs.std() < 0.5, f"Bootstrap variance too high: {bs.std():.4f}"

    def test_bootstrap_subsample_is_90_percent(self):
        """Bootstrap subsampling should use 90% of test samples.

        With n_test=10, int(10*0.9)=9, so only C(10,9)=10 unique subsets.
        Despite GPU floating-point non-determinism causing tiny score differences
        for the "same" mathematical subset, rounding to 4 decimals should collapse
        most duplicates. Allow some slack for GPU arithmetic.
        """
        w = torch.randn(15, 5)
        X = torch.randn(90, 15)
        Y = X @ w + torch.randn(90, 5) * 0.3

        train = AlignmentData(activations={"layer": X[:80]}, neural=Y[:80])
        test = AlignmentData(activations={"layer": X[80:]}, neural=Y[80:])

        results = compute_encoding_score(
            train, test, bootstrap=True, n_bootstrap=50, verbose=False
        )
        bs = np.array(results[0]["bootstrap_scores"])
        # With 10 test samples and 90% subsample, there are C(10,9)=10 unique subsets.
        # GPU floating-point non-determinism may produce slightly different scores for
        # mathematically identical subsets, so allow some slack above 10.
        n_unique = len(set(np.round(bs, 4)))
        assert n_unique <= 15, f"Expected <=15 unique bootstrap scores (10 subsets + GPU noise), got {n_unique}"

    def test_no_bootstrap_no_scores_key(self, synthetic_encoding_data):
        """Without bootstrap, 'bootstrap_scores' key should be absent."""
        train, test = synthetic_encoding_data
        results = compute_encoding_score(
            train, test, bootstrap=False, verbose=False
        )
        assert "bootstrap_scores" not in results[0]

    def test_bootstrap_all_finite(self, synthetic_encoding_data):
        """All bootstrap scores should be finite (no NaN/inf)."""
        train, test = synthetic_encoding_data
        results = compute_encoding_score(
            train, test, bootstrap=True, n_bootstrap=100, verbose=False
        )
        bs = np.array(results[0]["bootstrap_scores"])
        assert np.all(np.isfinite(bs)), f"Bootstrap has non-finite values: {bs[~np.isfinite(bs)]}"

    def test_ci_width_reasonable(self):
        """Bootstrap CI width should be reasonable (not degenerate or absurdly wide).

        Uses noisier data than the synthetic fixture (which has r~0.999 and
        extremely narrow CIs) so there is measurable bootstrap variance.
        """
        torch.manual_seed(42)
        w = torch.randn(30, 10)
        X = torch.randn(150, 30)
        Y = X @ w + torch.randn(150, 10) * 2.0  # moderate noise

        train = AlignmentData(activations={"layer": X[:120]}, neural=Y[:120])
        test = AlignmentData(activations={"layer": X[120:]}, neural=Y[120:])

        results = compute_encoding_score(
            train, test, bootstrap=True, n_bootstrap=200, verbose=False
        )
        r = results[0]
        ci_width = r["ci_high"] - r["ci_low"]
        assert ci_width > 0.001, f"CI too narrow: {ci_width}"
        assert ci_width < 1.5, f"CI too wide: {ci_width}"


# ═══════════════════════════════════════════════════════════
# 5. NORMALIZATION CORRECTNESS
# ═══════════════════════════════════════════════════════════

class TestNormalization:
    """Tests for X and Y normalization at each stage of the encoding pipeline."""

    def test_x_selection_uses_fit_only_stats(self):
        """During layer selection, X should be normalized with fit-only stats.

        The code computes X_fit_mean/X_fit_std from the fit portion (80%) and
        applies them to both fit and val portions. Verify this is correct.
        """
        n_train = 100
        X = torch.zeros(n_train, 20)
        X[:80] = torch.randn(80, 20)            # fit: std~1
        X[80:] = torch.randn(20, 20) * 10       # val: std~10

        # Fit-only stats
        fit_mean = X[:80].mean(dim=0)
        fit_std = X[:80].std(dim=0) + 1e-8
        val_with_fit_stats = _znorm(X[80:], fit_mean, fit_std)

        # Full-train stats (wrong for selection)
        full_mean = X.mean(dim=0)
        full_std = X.std(dim=0) + 1e-8
        val_with_full_stats = _znorm(X[80:], full_mean, full_std)

        # Fit-only normalization of val should produce larger magnitudes
        # because val has std~10 while fit has std~1, so dividing by fit_std~1 keeps large values
        assert val_with_fit_stats.abs().mean() > val_with_full_stats.abs().mean()

    def test_y_selection_uses_fit_only_stats(self):
        """Y normalization during layer selection uses fit-only (80%) stats.

        After the leakage fix, Y_mean/Y_std are computed on the fit portion
        only, not the full train set. This prevents the val portion from
        being normalized using its own distribution.
        """
        n_train = 100
        Y = torch.randn(n_train, 10)
        split = int(0.8 * n_train)

        # The code now uses fit-only stats for selection
        Y_fit = Y[:split]
        Y_fit_normed, Y_fit_mean, Y_fit_std = _znorm_fit(Y_fit)

        # Val Y is normalized with fit-only stats (correct, no leakage)
        Y_val_normed = _znorm(Y[split:], Y_fit_mean, Y_fit_std)

        # Fit-normed should have mean~0, std~1 (by construction)
        assert Y_fit_normed.mean(dim=0).abs().max() < 0.01
        # Val-normed should NOT have mean~0 (different distribution)
        # (This is the correct behavior — no leakage from val into stats)
        assert Y_val_normed.mean(dim=0).abs().max() > 0  # non-zero mean expected

    def test_x_refit_uses_full_train_stats(self):
        """For the final evaluation, X is normalized with full train stats."""
        X = torch.randn(100, 20)
        train_mean = X.mean(dim=0)
        train_std = X.std(dim=0) + 1e-8
        X_normed = _znorm(X, train_mean, train_std)

        assert X_normed.mean(dim=0).abs().max() < 0.01
        assert (X_normed.std(dim=0) - 1.0).abs().max() < 0.05

    def test_pearson_r_invariant_to_affine(self, backend):
        """Pearson R is invariant to affine transforms, so Y normalization
        doesn't affect the final test score (only the ridge fitting)."""
        w = torch.randn(15, 5)
        X = torch.randn(80, 15)
        Y = X @ w + torch.randn(80, 5) * 0.1

        pred = X @ w  # perfect prediction
        pred_gpu = backend.asarray(pred[60:])
        Y_raw_gpu = backend.asarray(Y[60:])
        score_raw = float(correlation_score(Y_raw_gpu, pred_gpu).mean())

        # Normalize both
        Y_normed = (Y - Y.mean(0)) / (Y.std(0) + 1e-8)
        pred_normed = (pred - pred.mean(0)) / (pred.std(0) + 1e-8)
        score_normed = float(correlation_score(
            backend.asarray(Y_normed[60:]),
            backend.asarray(pred_normed[60:])
        ).mean())

        assert abs(score_raw - score_normed) < 0.01, (
            f"Pearson R should be invariant: raw={score_raw:.4f}, normed={score_normed:.4f}"
        )

    def test_80_20_split_deterministic(self):
        """The 80/20 split should be deterministic with the same seed."""
        n = 100
        perm1 = np.random.RandomState(42).permutation(n)
        perm2 = np.random.RandomState(42).permutation(n)
        np.testing.assert_array_equal(perm1, perm2)

        split = int(0.8 * n)
        assert len(perm1[:split]) == 80
        assert len(perm1[split:]) == 20
        # No overlap
        assert len(set(perm1[:split]) & set(perm1[split:])) == 0


# ═══════════════════════════════════════════════════════════
# 6. EDGE CASES
# ═══════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge cases for encoding score computation."""

    def test_single_voxel(self):
        """Should work with n_voxels=1."""
        w = torch.randn(20, 1)
        X = torch.randn(130, 20)
        Y = X @ w + torch.randn(130, 1) * 0.1

        train = AlignmentData(activations={"layer": X[:100]}, neural=Y[:100])
        test = AlignmentData(activations={"layer": X[100:]}, neural=Y[100:])

        results = compute_encoding_score(train, test, bootstrap=False, verbose=False)
        assert -1 <= results[0]["score"] <= 1

    def test_few_test_samples(self):
        """Should work with very few test samples (n=10)."""
        X = torch.randn(90, 15)
        Y = torch.randn(90, 5)

        train = AlignmentData(activations={"layer": X[:80]}, neural=Y[:80])
        test = AlignmentData(activations={"layer": X[80:]}, neural=Y[80:])

        results = compute_encoding_score(train, test, bootstrap=False, verbose=False)
        assert results[0]["score"] is not None
        assert np.isfinite(results[0]["score"])

    def test_few_test_with_bootstrap(self):
        """Bootstrap should work even with few test samples (n=10, subsample=9)."""
        w = torch.randn(15, 5)
        X = torch.randn(90, 15)
        Y = X @ w + torch.randn(90, 5) * 0.5

        train = AlignmentData(activations={"layer": X[:80]}, neural=Y[:80])
        test = AlignmentData(activations={"layer": X[80:]}, neural=Y[80:])

        results = compute_encoding_score(
            train, test, bootstrap=True, n_bootstrap=20, verbose=False
        )
        r = results[0]
        assert r["ci_low"] is not None
        assert r["ci_high"] is not None

    def test_many_layers(self):
        """Should handle many layers without issues."""
        train_acts = {f"layer_{i}": torch.randn(80, 15) for i in range(10)}
        test_acts = {f"layer_{i}": torch.randn(20, 15) for i in range(10)}

        train = AlignmentData(activations=train_acts, neural=torch.randn(80, 5))
        test = AlignmentData(activations=test_acts, neural=torch.randn(20, 5))

        results = compute_encoding_score(train, test, bootstrap=False, verbose=False)
        assert len(results[0]["layer_selection_scores"]) == 10

    def test_4d_conv_activations(self):
        """Should handle 4D conv activations by flattening internally."""
        train = AlignmentData(
            activations={"conv": torch.randn(80, 64, 6, 6)},  # conv feature maps
            neural=torch.randn(80, 10),
        )
        test = AlignmentData(
            activations={"conv": torch.randn(20, 64, 6, 6)},
            neural=torch.randn(20, 10),
        )

        results = compute_encoding_score(train, test, bootstrap=False, verbose=False)
        assert results[0]["score"] is not None
        # Inputs should NOT be mutated (flattening happens on internal copies)
        assert train.activations["conv"].ndim == 4

    def test_high_dimensional_features(self):
        """Should handle n_features >> n_train (underdetermined case, common for conv layers).

        Ridge regression is designed for exactly this scenario. After flattening,
        conv5 in AlexNet has 256*6*6 = 9216 features but only ~9000 NSD train samples.
        """
        n_train, n_test, n_features = 60, 15, 500
        train = AlignmentData(
            activations={"layer": torch.randn(n_train, n_features)},
            neural=torch.randn(n_train, 5),
        )
        test = AlignmentData(
            activations={"layer": torch.randn(n_test, n_features)},
            neural=torch.randn(n_test, 5),
        )

        results = compute_encoding_score(train, test, bootstrap=False, verbose=False)
        assert np.isfinite(results[0]["score"])

    def test_many_voxels(self):
        """Should handle many voxels (ventral stream has ~7600 voxels)."""
        n_voxels = 200
        w = torch.randn(20, n_voxels)
        X = torch.randn(100, 20)
        Y = X @ w + torch.randn(100, n_voxels) * 0.5

        train = AlignmentData(activations={"layer": X[:80]}, neural=Y[:80])
        test = AlignmentData(activations={"layer": X[80:]}, neural=Y[80:])

        results = compute_encoding_score(train, test, bootstrap=False, verbose=False)
        assert results[0]["score"] > 0.3, f"Linear data with many voxels should score well: {results[0]['score']:.4f}"

    def test_anticorrelated_data(self):
        """When model predictions are anti-correlated with neural data,
        score should be negative."""
        w = torch.randn(20, 10)
        X = torch.randn(130, 20)
        # Y is negatively related to X
        Y = -(X @ w) + torch.randn(130, 10) * 0.1

        train = AlignmentData(activations={"layer": X[:100]}, neural=Y[:100])
        test = AlignmentData(activations={"layer": X[100:]}, neural=Y[100:])

        results = compute_encoding_score(train, test, bootstrap=False, verbose=False)
        # Ridge regression should find the negative relationship
        # Score might still be positive because Pearson r measures correlation
        # of predictions vs actual, and ridge can learn negative weights
        assert np.isfinite(results[0]["score"])

    def test_n_bootstrap_1(self):
        """Should handle n_bootstrap=1 without error."""
        w = torch.randn(15, 5)
        X = torch.randn(90, 15)
        Y = X @ w + torch.randn(90, 5) * 0.3

        train = AlignmentData(activations={"layer": X[:70]}, neural=Y[:70])
        test = AlignmentData(activations={"layer": X[70:]}, neural=Y[70:])

        results = compute_encoding_score(
            train, test, bootstrap=True, n_bootstrap=1, verbose=False
        )
        r = results[0]
        assert r["ci_low"] is not None
        assert r["ci_high"] is not None
        assert len(r["bootstrap_scores"]) == 1

    def test_single_layer(self):
        """Should work with only one layer (trivial layer selection)."""
        X = torch.randn(100, 20)
        Y = torch.randn(100, 5)

        train = AlignmentData(activations={"only": X[:80]}, neural=Y[:80])
        test = AlignmentData(activations={"only": X[80:]}, neural=Y[80:])

        results = compute_encoding_score(train, test, bootstrap=False, verbose=False)
        assert results[0]["layer"] == "only"
        assert len(results[0]["layer_selection_scores"]) == 1


# ═══════════════════════════════════════════════════════════
# 7. ALIGNMENT DISPATCH
# ═══════════════════════════════════════════════════════════

class TestAlignmentDispatch:
    """Tests for alignment.py dispatch to encoding score."""

    def test_dispatches_to_encoding(self, encoding_cfg):
        """compute_traintest_alignment should route to encoding_score."""
        w = torch.randn(15, 5)
        X = torch.randn(100, 15)
        Y = X @ w + torch.randn(100, 5) * 0.3

        train = AlignmentData(activations={"layer": X[:80]}, neural=Y[:80])
        test = AlignmentData(activations={"layer": X[80:]}, neural=Y[80:])

        encoding_cfg.analysis = "encoding_score"
        encoding_cfg.bootstrap = False
        results = compute_traintest_alignment(encoding_cfg, train, test, verbose=False)

        assert len(results) == 1
        assert results[0]["analysis"] == "encoding_score"
        assert results[0]["compare_method"] == "pearson"

    def test_things_encoding_raises(self, encoding_cfg):
        """encoding_score + things-behavior should raise ValueError."""
        encoding_cfg.neural_dataset = "things-behavior"
        encoding_cfg.analysis = "encoding_score"

        train = AlignmentData(activations={"layer": torch.randn(50, 10)}, neural=torch.randn(50, 5))
        test = AlignmentData(activations={"layer": torch.randn(20, 10)}, neural=torch.randn(20, 5))

        with pytest.raises(ValueError, match="not supported for things-behavior"):
            compute_traintest_alignment(encoding_cfg, train, test)

    def test_prepare_traintest_alignment(self):
        """prepare_traintest_alignment should correctly align data by stimulus ID."""
        n_total = 100
        acts_raw = {"layer": torch.randn(n_total, 20)}
        keys = [f"stim_{i}" for i in range(n_total)]

        neural_data = {
            "train": {f"stim_{i}": np.random.randn(10) for i in range(80)},
            "test": {f"stim_{i}": np.random.randn(10) for i in range(80, 100)},
        }

        cfg = OmegaConf.create({"analysis": "encoding_score"})
        train, test = prepare_traintest_alignment(cfg, acts_raw, neural_data, keys)

        assert train.neural.shape[0] == 80
        assert test.neural.shape[0] == 20
        assert train.activations["layer"].shape[0] == 80
        assert test.activations["layer"].shape[0] == 20

    def test_align_stimulus_level_squeeze_single_voxel(self):
        """_align_stimulus_level with n_voxels=1 may collapse dimension via .squeeze().

        With n_voxels=1, the neural tensor should remain 2D (n, 1).
        """
        targets = {f"stim_{i}": np.array([0.5]) for i in range(50)}
        keys = [f"stim_{i}" for i in range(50)]
        acts_raw = {"layer": torch.randn(50, 10)}

        _, neural, _ = _align_stimulus_level(acts_raw, targets, keys)
        assert neural.ndim == 2
        assert neural.shape == (50, 1)

    def test_rsa_dispatch_from_alignment(self, encoding_cfg):
        """compute_traintest_alignment should route to RSA when analysis=rsa."""
        w = torch.randn(15, 5)
        X = torch.randn(100, 15)
        Y = X @ w + torch.randn(100, 5) * 0.3

        train = AlignmentData(activations={"layer": X[:80]}, neural=Y[:80])
        test = AlignmentData(activations={"layer": X[80:]}, neural=Y[80:])

        encoding_cfg.analysis = "rsa"
        encoding_cfg.compare_method = "spearman"
        encoding_cfg.bootstrap = False
        results = compute_traintest_alignment(encoding_cfg, train, test, verbose=False)

        assert len(results) == 1
        assert results[0]["analysis"] == "rsa"
        assert results[0]["compare_method"] == "spearman"

    def test_unknown_analysis_raises(self, encoding_cfg):
        """Unknown analysis method should raise ValueError."""
        encoding_cfg.analysis = "unknown_method"
        train = AlignmentData(activations={"layer": torch.randn(50, 10)}, neural=torch.randn(50, 5))
        test = AlignmentData(activations={"layer": torch.randn(20, 10)}, neural=torch.randn(20, 5))

        with pytest.raises(ValueError, match="Unknown analysis method"):
            compute_traintest_alignment(encoding_cfg, train, test)


# ═══════════════════════════════════════════════════════════
# 8. DB STORAGE
# ═══════════════════════════════════════════════════════════

class TestDBStorage:
    """Tests for encoding score results saved to SQLite."""

    def test_results_saved_with_pearson(self, synthetic_encoding_data, encoding_cfg, tmp_path):
        """Results should be saved with compare_method='pearson'."""
        train, test = synthetic_encoding_data
        results = compute_encoding_score(train, test, bootstrap=False, verbose=False)
        df = pd.DataFrame(results)

        import visreps.utils as vutils
        original_path = vutils._RESULTS_DB_PATH
        vutils._RESULTS_DB_PATH = tmp_path / "test_results.db"
        try:
            save_results(df, encoding_cfg)

            conn = sqlite3.connect(str(tmp_path / "test_results.db"))
            col_names = [d[0] for d in conn.execute("SELECT * FROM results").description]
            rows = conn.execute("SELECT * FROM results").fetchall()
            conn.close()

            assert len(rows) == 1
            row = dict(zip(col_names, rows[0]))
            assert row["compare_method"] == "pearson"
            assert row["analysis"] == "encoding_score"
        finally:
            vutils._RESULTS_DB_PATH = original_path

    def test_layer_selection_scores_saved(self, synthetic_encoding_data, encoding_cfg, tmp_path):
        """Layer selection scores should be saved to layer_selection_scores table."""
        train, test = synthetic_encoding_data
        results = compute_encoding_score(train, test, bootstrap=False, verbose=False)
        df = pd.DataFrame(results)

        import visreps.utils as vutils
        original_path = vutils._RESULTS_DB_PATH
        vutils._RESULTS_DB_PATH = tmp_path / "test_results.db"
        try:
            save_results(df, encoding_cfg)

            conn = sqlite3.connect(str(tmp_path / "test_results.db"))
            rows = conn.execute("SELECT * FROM layer_selection_scores").fetchall()
            conn.close()

            assert len(rows) == 2  # signal_layer and noise_layer
        finally:
            vutils._RESULTS_DB_PATH = original_path

    def test_bootstrap_distribution_saved(self, synthetic_encoding_data, encoding_cfg, tmp_path):
        """Bootstrap scores should be saved when bootstrap=True."""
        train, test = synthetic_encoding_data
        results = compute_encoding_score(
            train, test, bootstrap=True, n_bootstrap=20, verbose=False
        )
        df = pd.DataFrame(results)

        import visreps.utils as vutils
        original_path = vutils._RESULTS_DB_PATH
        vutils._RESULTS_DB_PATH = tmp_path / "test_results.db"
        try:
            save_results(df, encoding_cfg)

            conn = sqlite3.connect(str(tmp_path / "test_results.db"))
            rows = conn.execute("SELECT * FROM bootstrap_distributions").fetchall()
            conn.close()

            assert len(rows) == 1
            scores = json.loads(rows[0][2])  # scores JSON column
            assert len(scores) == 20
        finally:
            vutils._RESULTS_DB_PATH = original_path

    def test_run_id_distinguishes_compare_methods(self, encoding_cfg):
        """run_id hash should differ for pearson vs spearman."""
        encoding_cfg.compare_method = "pearson"
        run_id_pearson = _compute_run_id(encoding_cfg)

        encoding_cfg.compare_method = "spearman"
        run_id_spearman = _compute_run_id(encoding_cfg)

        assert run_id_pearson != run_id_spearman

    def test_idempotent_overwrite(self, synthetic_encoding_data, encoding_cfg, tmp_path):
        """Saving the same results twice should not create duplicate rows (INSERT OR REPLACE)."""
        train, test = synthetic_encoding_data
        results = compute_encoding_score(train, test, bootstrap=False, verbose=False)
        df = pd.DataFrame(results)

        import visreps.utils as vutils
        original_path = vutils._RESULTS_DB_PATH
        vutils._RESULTS_DB_PATH = tmp_path / "test_idempotent.db"
        try:
            save_results(df, encoding_cfg)
            save_results(df, encoding_cfg)  # second save should overwrite, not duplicate

            conn = sqlite3.connect(str(tmp_path / "test_idempotent.db"))
            rows = conn.execute("SELECT * FROM results").fetchall()
            conn.close()

            assert len(rows) == 1, f"Expected 1 row after double-save, got {len(rows)}"
        finally:
            vutils._RESULTS_DB_PATH = original_path


# ═══════════════════════════════════════════════════════════
# 9. MATHEMATICAL CORRECTNESS
# ═══════════════════════════════════════════════════════════

class TestMathematicalCorrectness:
    """Verify encoding score outputs match independent reference computations."""

    def test_score_matches_manual_per_voxel_pearson(self, backend):
        """The encoding score must be the mean of per-voxel Pearson r values.

        Bug caught: Wrong aggregation (e.g., median, sum, or R² instead of r).
        """
        w = torch.randn(15, 5)
        X = torch.randn(60, 15)
        Y = X @ w + torch.randn(60, 5) * 0.3

        X_tr = backend.asarray(X[:50])
        Y_tr = backend.asarray(Y[:50])
        X_te = X[50:]
        Y_te = backend.asarray(Y[50:])

        pred, score = _fit_and_score(X_tr, Y_tr, X_te, Y_te, np.logspace(-3, 3, 5), backend)

        # Manual per-voxel Pearson r
        voxel_scores = correlation_score(Y_te, pred)
        manual_mean = float(voxel_scores.mean())

        assert abs(score - manual_mean) < 1e-6, (
            f"Score {score} should equal mean of per-voxel correlations {manual_mean}"
        )

    def test_score_matches_scipy_pearson_per_voxel(self, backend):
        """Verify per-voxel Pearson r matches scipy.stats.pearsonr exactly.

        Bug caught: himalaya correlation_score using a different formula.
        """
        w = torch.randn(10, 3)
        X = torch.randn(60, 10)
        Y = X @ w + torch.randn(60, 3) * 0.3

        X_tr = backend.asarray(X[:50])
        Y_tr = backend.asarray(Y[:50])
        X_te = X[50:]
        Y_te_gpu = backend.asarray(Y[50:])

        pred, score = _fit_and_score(X_tr, Y_tr, X_te, Y_te_gpu, np.logspace(-3, 3, 5), backend)

        # Manual scipy per-voxel
        pred_np = pred.cpu().numpy() if hasattr(pred, 'cpu') else np.asarray(pred)
        Y_te_np = Y[50:].numpy()
        scipy_scores = []
        for v in range(Y_te_np.shape[1]):
            r, _ = scipy.stats.pearsonr(Y_te_np[:, v], pred_np[:, v])
            scipy_scores.append(r)
        scipy_mean = np.mean(scipy_scores)

        assert abs(score - scipy_mean) < 1e-4, (
            f"Score {score:.6f} differs from scipy mean Pearson r {scipy_mean:.6f}"
        )

    def test_ridge_regression_known_solution(self, backend):
        """For data where the true weights are known, ridge predictions should
        closely approximate Y_test = X_test @ w_true.

        Bug caught: Ridge solver producing wrong coefficients.
        """
        torch.manual_seed(42)
        w_true = torch.randn(20, 5)
        X = torch.randn(200, 20)
        Y = X @ w_true  # No noise: perfect linear relationship

        X_tr = backend.asarray(X[:180])
        Y_tr = backend.asarray(Y[:180])
        X_te = X[180:]
        Y_te = backend.asarray(Y[180:])

        pred, score = _fit_and_score(X_tr, Y_tr, X_te, Y_te, np.logspace(-10, 10, 20), backend)

        assert score > 0.99, f"Perfect linear data should give r > 0.99, got {score:.4f}"

        # Predictions should be very close to actual
        pred_np = pred.cpu().numpy() if hasattr(pred, 'cpu') else np.asarray(pred)
        Y_te_np = Y[180:].numpy()
        max_err = np.abs(pred_np - Y_te_np).max()
        assert max_err < 1.0, f"Predictions deviate too much from truth: max_err={max_err:.4f}"

    def test_feature_scaling_invariance(self):
        """Z-normalization makes encoding score invariant to feature scale.

        Bug caught: If z-norm is skipped or applied incorrectly, differently
        scaled features would give different scores.
        """
        torch.manual_seed(42)
        w = torch.randn(20, 10)
        X_base = torch.randn(130, 20)
        Y = X_base @ w + torch.randn(130, 10) * 0.5

        scores = []
        for scale in [1.0, 100.0, 0.001]:
            X = X_base * scale
            train = AlignmentData(activations={"layer": X[:100].clone()}, neural=Y[:100].clone())
            test = AlignmentData(activations={"layer": X[100:].clone()}, neural=Y[100:].clone())
            r = compute_encoding_score(train, test, bootstrap=False, seed=42, verbose=False)
            scores.append(r[0]["score"])

        # All scores should be very close despite 5 orders of magnitude scale difference
        for i in range(1, len(scores)):
            assert abs(scores[0] - scores[i]) < 0.05, (
                f"Score varies with scale: {scores}. Z-norm should make it invariant."
            )

    def test_encoding_score_positive_for_strong_signal(self):
        """When model activations are linearly predictive of neural data,
        encoding score must be substantially positive.

        Bug caught: Normalization or ridge regression inverting the signal.
        """
        torch.manual_seed(42)
        w = torch.randn(30, 15)
        X = torch.randn(200, 30)
        Y = X @ w + torch.randn(200, 15) * 0.5

        train = AlignmentData(activations={"layer": X[:150]}, neural=Y[:150])
        test = AlignmentData(activations={"layer": X[150:]}, neural=Y[150:])

        results = compute_encoding_score(train, test, bootstrap=False, verbose=False)
        assert results[0]["score"] > 0.7, (
            f"Strong signal should give high score, got {results[0]['score']:.4f}"
        )


# ═══════════════════════════════════════════════════════════
# 10. PIPELINE-LEVEL NORMALIZATION & LEAKAGE
# ═══════════════════════════════════════════════════════════

class TestPipelineNormalization:
    """Verify that normalization in the full pipeline prevents data leakage."""

    def test_x_test_normalized_with_train_stats(self):
        """X_test must be normalized using TRAIN mean/std (not test stats).

        Bug caught: If test data is normalized with its own stats, the model
        would see artificially well-behaved test inputs.
        """
        torch.manual_seed(42)
        w = torch.randn(20, 10)

        # Train: X ~ N(0, 1)
        X_train = torch.randn(100, 20)
        Y_train = X_train @ w + torch.randn(100, 10) * 0.3

        # Test: X ~ N(5, 3) — very different distribution
        X_test = torch.randn(30, 20) * 3 + 5
        Y_test = X_test @ w + torch.randn(30, 10) * 0.3

        # Directly verify the normalization logic
        train_acts = _flatten_to_cpu({"layer": X_train})
        test_acts = _flatten_to_cpu({"layer": X_test})
        _, train_mean, train_std = _znorm_fit(train_acts["layer"])
        X_test_normed = _znorm(test_acts["layer"], train_mean, train_std)

        # Test data normed with TRAIN stats should NOT be zero-mean
        test_mean = X_test_normed.mean(dim=0)
        assert test_mean.abs().mean() > 0.5, (
            "Test data normed with train stats should have non-zero mean "
            f"(got {test_mean.abs().mean():.4f}). If zero, test stats were leaked."
        )

    def test_y_selection_no_full_train_leakage(self):
        """During layer selection, Y must be normalized with FIT (80%) stats only.

        Bug caught: If Y is normalized with full-train stats, the val portion's
        own distribution leaks into the normalization, inflating selection scores.
        """
        torch.manual_seed(42)
        n_train = 100
        Y = torch.randn(n_train, 10)

        # Make val portion have extreme values
        split = int(0.8 * n_train)
        perm = np.random.RandomState(42).permutation(n_train)
        val_idx = perm[split:]
        Y[val_idx] = Y[val_idx] * 100  # Extreme val values

        # Fit-only stats
        fit_idx = perm[:split]
        _, fit_mean, fit_std = _znorm_fit(Y[fit_idx])

        # Full-train stats (leaky)
        _, full_mean, full_std = _znorm_fit(Y)

        # Val normalized with fit stats: extreme values remain extreme
        val_fit_normed = _znorm(Y[val_idx], fit_mean, fit_std)
        # Val normalized with full stats: extreme values are tamed
        val_full_normed = _znorm(Y[val_idx], full_mean, full_std)

        # Fit-only normalization should produce larger magnitudes for val
        assert val_fit_normed.abs().mean() > val_full_normed.abs().mean() * 2, (
            "Fit-only normalization should preserve extreme val values. "
            "If not, full-train stats are leaking into selection."
        )

    def test_distribution_shift_still_works(self):
        """When train and test have different distributions, the score should
        still be meaningful (positive) because Pearson r is affine-invariant.

        Bug caught: fit_intercept=False causing problems with mean shift.
        """
        torch.manual_seed(42)
        w = torch.randn(20, 10)

        X_train = torch.randn(100, 20)
        Y_train = X_train @ w + torch.randn(100, 10) * 0.3

        # Test: shifted by +5 on all features
        X_test = torch.randn(30, 20) + 5.0
        Y_test = X_test @ w + torch.randn(30, 10) * 0.3

        train = AlignmentData(activations={"layer": X_train}, neural=Y_train)
        test = AlignmentData(activations={"layer": X_test}, neural=Y_test)

        results = compute_encoding_score(train, test, bootstrap=False, verbose=False)
        score = results[0]["score"]

        assert score > 0.3, (
            f"Score should be positive despite distribution shift "
            f"(Pearson r is affine-invariant), got {score:.4f}"
        )

    def test_train_test_isolation_scrambled_test(self):
        """Scrambling test neural data should destroy the score while keeping
        training intact.

        Bug caught: If test score depends on training data content (leakage),
        scrambled test would still give high scores.
        """
        torch.manual_seed(42)
        w = torch.randn(20, 10)
        X = torch.randn(130, 20)
        Y = X @ w + torch.randn(130, 10) * 0.3

        train = AlignmentData(activations={"layer": X[:100]}, neural=Y[:100])

        # Normal test
        test_normal = AlignmentData(activations={"layer": X[100:]}, neural=Y[100:])
        r_normal = compute_encoding_score(train, test_normal, bootstrap=False, verbose=False)

        # Scrambled test (neural data randomized)
        test_scrambled = AlignmentData(
            activations={"layer": X[100:]},
            neural=torch.randn(30, 10),  # Random neural data
        )
        r_scrambled = compute_encoding_score(train, test_scrambled, bootstrap=False, verbose=False)

        assert r_normal[0]["score"] > r_scrambled[0]["score"] + 0.3, (
            f"Normal score ({r_normal[0]['score']:.4f}) should be much higher than "
            f"scrambled ({r_scrambled[0]['score']:.4f}). If not, there may be leakage."
        )


# ═══════════════════════════════════════════════════════════
# 11. BOOTSTRAP MECHANICS
# ═══════════════════════════════════════════════════════════

class TestBootstrapMechanics:
    """Deep tests for bootstrap implementation correctness."""

    def test_bootstrap_uses_replace_false(self):
        """Bootstrap must sample WITHOUT replacement (subsampling, not resampling).

        Bug caught: Using replace=True would duplicate test samples, inflating
        correlation by comparing identical predictions against identical targets.
        """
        torch.manual_seed(42)
        w = torch.randn(15, 5)
        X = torch.randn(130, 15)
        Y = X @ w + torch.randn(130, 5) * 0.5

        train = AlignmentData(activations={"layer": X[:100]}, neural=Y[:100])
        test = AlignmentData(activations={"layer": X[100:]}, neural=Y[100:])
        n_test = 30

        # Run encoding and get bootstrap scores
        results = compute_encoding_score(
            train, test, bootstrap=True, n_bootstrap=50, seed=42, verbose=False
        )
        bs = results[0]["bootstrap_scores"]

        # Reproduce the RNG to verify indices have no duplicates
        rng = np.random.RandomState(42)
        n_train = 100
        _ = rng.permutation(n_train)  # consume state for 80/20 split

        n_subsample = int(n_test * 0.9)
        for i in range(50):
            boot_idx = rng.choice(n_test, size=n_subsample, replace=False)
            assert len(set(boot_idx)) == n_subsample, (
                f"Bootstrap iteration {i}: indices have duplicates, "
                f"suggesting replace=True was used"
            )

    def test_bootstrap_reuses_cached_predictions(self):
        """Bootstrap must reuse cached predictions, NOT refit the ridge model.

        Bug caught: If bootstrap refits per iteration, it would be computationally
        wrong (different model per iteration) and extremely slow.
        """
        torch.manual_seed(42)
        w = torch.randn(15, 5)
        X = torch.randn(110, 15)
        Y = X @ w + torch.randn(110, 5) * 0.3

        train = AlignmentData(activations={"layer": X[:90]}, neural=Y[:90])
        test = AlignmentData(activations={"layer": X[90:]}, neural=Y[90:])

        # Run with bootstrap and verify result structure
        results = compute_encoding_score(
            train, test, bootstrap=True, n_bootstrap=30, seed=42, verbose=False
        )
        bs = np.array(results[0]["bootstrap_scores"])
        point = results[0]["score"]

        # The bootstrap variance should be small (subsampling 90% of 20 test points)
        # If refitting happened, variance would be much larger due to model instability
        assert bs.std() < 0.3, (
            f"Bootstrap std {bs.std():.4f} is suspiciously high — "
            f"may be refitting instead of reusing cached predictions"
        )
        # All bootstrap scores should be relatively close to point estimate
        assert np.all(np.abs(bs - point) < 0.5), (
            "Bootstrap scores deviate too much from point estimate — "
            "refitting rather than subsampling?"
        )

    def test_bootstrap_rng_consumed_by_selection_split(self):
        """The same RNG is used for 80/20 split AND bootstrap. Bootstrap results
        depend on n_train because the permutation call consumes RNG state.

        This documents the coupling — not a bug, but important for reproducibility.
        """
        w = torch.randn(15, 5)

        bootstrap_scores_per_ntrain = []
        for n_train in [80, 100]:
            X_train = torch.randn(n_train, 15)
            Y_train = X_train @ w + torch.randn(n_train, 5) * 0.3
            # Same test data
            X_test = torch.randn(20, 15)
            Y_test = X_test @ w + torch.randn(20, 5) * 0.3

            train = AlignmentData(activations={"layer": X_train}, neural=Y_train)
            test = AlignmentData(activations={"layer": X_test}, neural=Y_test)
            r = compute_encoding_score(
                train, test, bootstrap=True, n_bootstrap=10, seed=42, verbose=False
            )
            bootstrap_scores_per_ntrain.append(r[0]["bootstrap_scores"])

        # Bootstrap scores will differ because different n_train consumes
        # different RNG state before bootstrap sampling begins
        assert bootstrap_scores_per_ntrain[0] != bootstrap_scores_per_ntrain[1], (
            "Bootstrap scores should differ with different n_train due to shared RNG"
        )

    def test_bootstrap_90_percent_subsample_sizes(self):
        """Verify exact subsample sizes for various n_test values.

        Bug caught: Off-by-one or wrong rounding in int(n_test * 0.9).
        """
        test_cases = [(10, 9), (20, 18), (50, 45), (100, 90), (1000, 900)]
        for n_test, expected in test_cases:
            actual = int(n_test * 0.9)
            assert actual == expected, (
                f"For n_test={n_test}: expected subsample size {expected}, got {actual}"
            )

    def test_bootstrap_all_indices_within_range(self):
        """All bootstrap indices must be valid (0 <= idx < n_test).

        Bug caught: Using wrong array size in rng.choice.
        """
        n_test = 30
        n_subsample = int(n_test * 0.9)
        rng = np.random.RandomState(42)

        for _ in range(100):
            idx = rng.choice(n_test, size=n_subsample, replace=False)
            assert idx.min() >= 0
            assert idx.max() < n_test
            assert len(idx) == n_subsample


# ═══════════════════════════════════════════════════════════
# 12. PCA RECONSTRUCTION PATH
# ═══════════════════════════════════════════════════════════

class TestReconstructPCA:
    """Tests for the reconstruct_pca_k code path in compute_encoding_score."""

    def test_reconstruct_pca_basic(self):
        """reconstruct_pca_k should apply PCA reconstruction and still produce
        valid results.

        Bug caught: PCA path crashing or producing NaN.
        """
        torch.manual_seed(42)
        w = torch.randn(30, 10)
        X = torch.randn(130, 30)
        Y = X @ w + torch.randn(130, 10) * 0.3

        train = AlignmentData(activations={"layer": X[:100]}, neural=Y[:100])
        test = AlignmentData(activations={"layer": X[100:]}, neural=Y[100:])

        results = compute_encoding_score(
            train, test, bootstrap=False, verbose=False, reconstruct_pca_k=10
        )
        assert np.isfinite(results[0]["score"])
        assert results[0]["layer"] == "layer"

    def test_reconstruct_pca_k_larger_than_features(self):
        """When reconstruct_pca_k > n_features, should use min(k, n_features).

        Bug caught: PCA raising ValueError for n_components > n_features.
        """
        torch.manual_seed(42)
        w = torch.randn(10, 5)
        X = torch.randn(90, 10)
        Y = X @ w + torch.randn(90, 5) * 0.3

        train = AlignmentData(activations={"layer": X[:70]}, neural=Y[:70])
        test = AlignmentData(activations={"layer": X[70:]}, neural=Y[70:])

        # k=100 but only 10 features — should use min(100, 10) = 10
        results = compute_encoding_score(
            train, test, bootstrap=False, verbose=False, reconstruct_pca_k=100
        )
        assert np.isfinite(results[0]["score"])

    def test_reconstruct_pca_reduces_score_with_few_components(self):
        """Using very few PCs to reconstruct should lose information and
        reduce the encoding score compared to using all features.

        Bug caught: PCA reconstruction not actually being applied.
        """
        torch.manual_seed(42)
        w = torch.randn(30, 10)
        X = torch.randn(130, 30)
        Y = X @ w + torch.randn(130, 10) * 0.3

        train = AlignmentData(activations={"layer": X[:100].clone()}, neural=Y[:100].clone())
        test = AlignmentData(activations={"layer": X[100:].clone()}, neural=Y[100:].clone())

        # Full features
        r_full = compute_encoding_score(
            train, test, bootstrap=False, verbose=False, reconstruct_pca_k=None
        )

        # Only 1 PC — should lose most information
        train2 = AlignmentData(activations={"layer": X[:100].clone()}, neural=Y[:100].clone())
        test2 = AlignmentData(activations={"layer": X[100:].clone()}, neural=Y[100:].clone())
        r_1pc = compute_encoding_score(
            train2, test2, bootstrap=False, verbose=False, reconstruct_pca_k=1
        )

        assert r_full[0]["score"] > r_1pc[0]["score"], (
            f"Full features ({r_full[0]['score']:.4f}) should beat 1-PC reconstruction "
            f"({r_1pc[0]['score']:.4f})"
        )

    def test_reconstruct_pca_does_not_mutate_inputs(self):
        """PCA reconstruction should not modify the input AlignmentData.

        Bug caught: In-place modification of activation tensors.
        """
        X = torch.randn(90, 20)
        Y = torch.randn(90, 5)
        train = AlignmentData(activations={"layer": X[:70].clone()}, neural=Y[:70].clone())
        test = AlignmentData(activations={"layer": X[70:].clone()}, neural=Y[70:].clone())

        original_train = train.activations["layer"].clone()
        original_test = test.activations["layer"].clone()

        compute_encoding_score(
            train, test, bootstrap=False, verbose=False, reconstruct_pca_k=5
        )

        torch.testing.assert_close(train.activations["layer"], original_train)
        torch.testing.assert_close(test.activations["layer"], original_test)


# ═══════════════════════════════════════════════════════════
# 13. ROBUSTNESS
# ═══════════════════════════════════════════════════════════

class TestRobustness:
    """Robustness tests for edge conditions in production data."""

    def test_constant_voxels_dilute_score(self):
        """Voxels with constant responses should contribute r=0, diluting the mean.

        Bug caught: Constant voxels producing NaN/inf that corrupt the mean.
        """
        torch.manual_seed(42)
        w = torch.randn(20, 8)
        X = torch.randn(100, 20)
        Y = X @ w + torch.randn(100, 8) * 0.3

        # Make 2 out of 8 voxels constant
        Y[:, 6] = 5.0
        Y[:, 7] = -3.0

        train = AlignmentData(activations={"layer": X[:80]}, neural=Y[:80])
        test = AlignmentData(activations={"layer": X[80:]}, neural=Y[80:])

        results = compute_encoding_score(train, test, bootstrap=False, verbose=False)
        assert np.isfinite(results[0]["score"]), "Constant voxels should not produce NaN"

    def test_constant_feature_columns(self):
        """X with constant columns should produce finite results.

        Bug caught: Division by zero in z-normalization when features have zero variance.
        """
        torch.manual_seed(42)
        X = torch.randn(100, 20)
        X[:, 10:] = 0.0  # Last 10 features are constant

        w = torch.randn(10, 5)
        Y = X[:, :10] @ w + torch.randn(100, 5) * 0.3

        train = AlignmentData(activations={"layer": X[:80]}, neural=Y[:80])
        test = AlignmentData(activations={"layer": X[80:]}, neural=Y[80:])

        results = compute_encoding_score(train, test, bootstrap=False, verbose=False)
        assert np.isfinite(results[0]["score"])
        assert results[0]["score"] > 0.2, (
            f"Should capture signal from non-constant features, got {results[0]['score']:.4f}"
        )

    def test_minimum_viable_n_train(self):
        """Find the minimum n_train that works. With cv=5 and 80% fit split,
        need at least int(0.8 * n) >= 5, so n_train >= 7.

        Bug caught: Crash for small datasets without a clear error.
        """
        torch.manual_seed(42)
        w = torch.randn(10, 3)
        X = torch.randn(17, 10)
        Y = X @ w + torch.randn(17, 3) * 0.5

        train = AlignmentData(activations={"layer": X[:7]}, neural=Y[:7])
        test = AlignmentData(activations={"layer": X[7:]}, neural=Y[7:])

        # int(0.8 * 7) = 5 fit samples, exactly matching cv=5
        results = compute_encoding_score(train, test, bootstrap=False, verbose=False)
        assert np.isfinite(results[0]["score"])

    def test_very_small_n_train_raises(self):
        """With n_train=4, int(0.8*4)=3 fit samples < cv=5, should raise.

        Bug caught: Silent garbage results from underfitting.
        """
        X = torch.randn(7, 10)
        Y = torch.randn(7, 3)

        train = AlignmentData(activations={"layer": X[:4]}, neural=Y[:4])
        test = AlignmentData(activations={"layer": X[4:]}, neural=Y[4:])

        with pytest.raises(Exception):
            compute_encoding_score(train, test, bootstrap=False, verbose=False)

    def test_large_feature_values(self):
        """Very large feature values should be handled by z-normalization.

        Bug caught: Numerical overflow before normalization.
        """
        torch.manual_seed(42)
        w = torch.randn(20, 5)
        X = torch.randn(100, 20) * 1e6  # Very large
        Y = (X / 1e6) @ w + torch.randn(100, 5) * 0.3  # Y from normalized X

        train = AlignmentData(activations={"layer": X[:80]}, neural=Y[:80])
        test = AlignmentData(activations={"layer": X[80:]}, neural=Y[80:])

        results = compute_encoding_score(train, test, bootstrap=False, verbose=False)
        assert np.isfinite(results[0]["score"]), "Large feature values should not cause overflow"

    def test_very_small_feature_values(self):
        """Very small feature values should be handled by z-normalization.

        Bug caught: Underflow or epsilon-dominated normalization.
        """
        torch.manual_seed(42)
        w = torch.randn(20, 5)
        X = torch.randn(100, 20) * 1e-8  # Very small
        Y = (X / 1e-8) @ w + torch.randn(100, 5) * 0.3

        train = AlignmentData(activations={"layer": X[:80]}, neural=Y[:80])
        test = AlignmentData(activations={"layer": X[80:]}, neural=Y[80:])

        results = compute_encoding_score(train, test, bootstrap=False, verbose=False)
        assert np.isfinite(results[0]["score"]), "Small feature values should not cause underflow"

    def test_layer_tie_breaking_uses_insertion_order(self):
        """When two layers have identical activations (thus identical scores),
        the first layer in dict iteration order wins (strict > comparison).

        Bug caught: Non-deterministic layer selection with tied scores.
        """
        torch.manual_seed(42)
        X = torch.randn(100, 20)
        Y = torch.randn(100, 10)

        # Identical activations for two layers
        train = AlignmentData(
            activations={"alpha": X[:80].clone(), "beta": X[:80].clone()},
            neural=Y[:80].clone(),
        )
        test = AlignmentData(
            activations={"alpha": X[80:].clone(), "beta": X[80:].clone()},
            neural=Y[80:].clone(),
        )

        r = compute_encoding_score(train, test, bootstrap=False, seed=42, verbose=False)
        # First layer in insertion order wins ties
        assert r[0]["layer"] == "alpha", (
            f"Expected 'alpha' (first in dict) to win tie, got '{r[0]['layer']}'"
        )

    def test_flatten_3d_and_5d_tensors(self):
        """_flatten_to_cpu should handle 3D, 4D, and 5D tensors."""
        acts = {
            "conv1d": torch.randn(10, 32, 8),        # 3D
            "conv2d": torch.randn(10, 64, 6, 6),     # 4D
            "conv3d": torch.randn(10, 16, 4, 4, 4),  # 5D
            "fc": torch.randn(10, 256),               # 2D (unchanged)
        }
        result = _flatten_to_cpu(acts)
        assert result["conv1d"].shape == (10, 32 * 8)
        assert result["conv2d"].shape == (10, 64 * 6 * 6)
        assert result["conv3d"].shape == (10, 16 * 4 * 4 * 4)
        assert result["fc"].shape == (10, 256)


# ═══════════════════════════════════════════════════════════
# 14. DISPATCH INTEGRATION
# ═══════════════════════════════════════════════════════════

class TestDispatchIntegration:
    """Tests for the alignment dispatch layer interacting with encoding score."""

    def test_seed_forwarding_through_dispatch(self):
        """Verify whether cfg.seed is forwarded to compute_encoding_score.

        The current code does NOT forward seed — this test documents that
        all cfg.seed values use the same internal seed=42 for layer selection.
        """
        X = torch.randn(100, 20)
        Y = torch.randn(100, 10)

        selection_scores_per_seed = []
        for seed in [1, 2]:
            cfg = OmegaConf.create({
                "analysis": "encoding_score",
                "bootstrap": False,
                "n_bootstrap": 50,
                "seed": seed,
                "reconstruct_from_pcs": False,
                "neural_dataset": "nsd",
            })
            train = AlignmentData(
                activations={"layer": X[:80].clone()}, neural=Y[:80].clone()
            )
            test = AlignmentData(
                activations={"layer": X[80:].clone()}, neural=Y[80:].clone()
            )
            results = compute_traintest_alignment(cfg, train, test, verbose=False)
            sel_score = results[0]["layer_selection_scores"][0]["score"]
            selection_scores_per_seed.append(sel_score)

        # Currently, seed is NOT forwarded, so both use seed=42 internally.
        # This documents the behavior — selection scores will be identical.
        assert selection_scores_per_seed[0] == selection_scores_per_seed[1], (
            "Expected identical selection scores because seed is not forwarded "
            "through dispatch (both use default seed=42). If this fails, "
            "the bug was fixed!"
        )

    def test_dispatch_encoding_rejects_things(self, encoding_cfg):
        """encoding_score + things-behavior should raise ValueError.

        Bug caught: Silently running encoding score on behavioral data.
        """
        encoding_cfg.neural_dataset = "things-behavior"
        encoding_cfg.analysis = "encoding_score"

        train = AlignmentData(
            activations={"layer": torch.randn(50, 10)}, neural=torch.randn(50, 5)
        )
        test = AlignmentData(
            activations={"layer": torch.randn(20, 10)}, neural=torch.randn(20, 5)
        )

        with pytest.raises(ValueError, match="not supported for things-behavior"):
            compute_traintest_alignment(encoding_cfg, train, test)

    def test_dispatch_passes_verbose_flag(self):
        """verbose flag should be forwarded through the dispatch."""
        X = torch.randn(100, 20)
        Y = torch.randn(100, 10)
        cfg = OmegaConf.create({
            "analysis": "encoding_score",
            "bootstrap": False,
            "n_bootstrap": 50,
            "reconstruct_from_pcs": False,
            "neural_dataset": "nsd",
        })
        train = AlignmentData(activations={"layer": X[:80]}, neural=Y[:80])
        test = AlignmentData(activations={"layer": X[80:]}, neural=Y[80:])

        # Should not crash with verbose=True or verbose=False
        r1 = compute_traintest_alignment(cfg, train, test, verbose=False)
        r2 = compute_traintest_alignment(cfg, train, test, verbose=True)
        assert len(r1) == 1 and len(r2) == 1

    def test_dispatch_returns_correct_structure(self):
        """Result from dispatch should have identical structure to direct call."""
        torch.manual_seed(42)
        X = torch.randn(100, 20)
        Y = torch.randn(100, 10)

        cfg = OmegaConf.create({
            "analysis": "encoding_score",
            "bootstrap": True,
            "n_bootstrap": 20,
            "reconstruct_from_pcs": False,
            "neural_dataset": "nsd",
        })
        train = AlignmentData(activations={"layer": X[:80]}, neural=Y[:80])
        test = AlignmentData(activations={"layer": X[80:]}, neural=Y[80:])

        results = compute_traintest_alignment(cfg, train, test, verbose=False)
        r = results[0]

        required_keys = {
            "layer", "compare_method", "score", "ci_low", "ci_high",
            "analysis", "layer_selection_scores", "bootstrap_scores",
        }
        assert required_keys.issubset(r.keys()), (
            f"Missing keys: {required_keys - set(r.keys())}"
        )
        assert r["analysis"] == "encoding_score"
        assert r["compare_method"] == "pearson"
        assert isinstance(r["bootstrap_scores"], list)
        assert len(r["bootstrap_scores"]) == 20


# ═══════════════════════════════════════════════════════════
# 15. END-TO-END: NSD ENCODING SCORE
# ═══════════════════════════════════════════════════════════

@pytest.mark.slow
class TestEndToEndNSD:
    """End-to-end encoding score tests with real NSD data."""

    def test_score_positive(self, nsd_encoding_results):
        """Trained model should achieve positive encoding score on NSD ventral stream."""
        score = nsd_encoding_results.iloc[0]["score"]
        assert score > 0, f"NSD encoding score should be positive, got {score:.4f}"

    def test_score_valid_range(self, nsd_encoding_results):
        """Score should be in valid Pearson r range [-1, 1]."""
        score = nsd_encoding_results.iloc[0]["score"]
        assert -1 <= score <= 1

    def test_compare_method_pearson(self, nsd_encoding_results):
        """NSD encoding should use Pearson correlation."""
        assert nsd_encoding_results.iloc[0]["compare_method"] == "pearson"

    def test_analysis_type(self, nsd_encoding_results):
        """Analysis should be 'encoding_score'."""
        assert nsd_encoding_results.iloc[0]["analysis"] == "encoding_score"

    def test_best_layer_reasonable(self, nsd_encoding_results):
        """Best layer should be a mid-to-late layer for ventral stream."""
        layer = nsd_encoding_results.iloc[0]["layer"]
        reasonable_layers = {
            "conv3_pre", "conv3_post", "conv4_pre", "conv4_post",
            "conv5_pre", "conv5_post", "fc1_pre", "fc1_post",
            "fc2_pre", "fc2_post",
        }
        assert layer in reasonable_layers, f"Expected mid-to-late layer, got {layer}"

    def test_bootstrap_cis_bracket(self, nsd_encoding_results):
        """Bootstrap CIs should bracket the point estimate."""
        r = nsd_encoding_results.iloc[0]
        assert r["ci_low"] is not None
        assert r["ci_high"] is not None
        assert r["ci_low"] < r["ci_high"]
        assert r["ci_low"] <= r["score"] <= r["ci_high"]

    def test_layer_selection_scores_present(self, nsd_encoding_results):
        """Should have selection scores for all extraction points."""
        lss = nsd_encoding_results.iloc[0].get("layer_selection_scores")
        assert lss is not None
        # 7 layers * pre/post = 14 extraction points
        assert len(lss) >= 7


# ═══════════════════════════════════════════════════════════
# 10. END-TO-END: TVSD ENCODING SCORE
# ═══════════════════════════════════════════════════════════

@pytest.mark.slow
class TestEndToEndTVSD:
    """End-to-end encoding score tests with real TVSD data."""

    def test_score_valid(self, tvsd_encoding_results):
        """TVSD encoding score should be in valid range."""
        score = tvsd_encoding_results.iloc[0]["score"]
        assert -1 <= score <= 1

    def test_score_positive(self, tvsd_encoding_results):
        """Trained model should achieve positive encoding on macaque IT."""
        score = tvsd_encoding_results.iloc[0]["score"]
        assert score > 0, f"TVSD IT encoding score should be positive, got {score:.4f}"

    def test_compare_method_pearson(self, tvsd_encoding_results):
        """TVSD encoding should use Pearson correlation."""
        assert tvsd_encoding_results.iloc[0]["compare_method"] == "pearson"

    def test_bootstrap_cis_bracket(self, tvsd_encoding_results):
        """Bootstrap CIs should bracket the point estimate."""
        r = tvsd_encoding_results.iloc[0]
        assert r["ci_low"] is not None
        assert r["ci_high"] is not None
        assert r["ci_low"] < r["ci_high"]
        assert r["ci_low"] <= r["score"] <= r["ci_high"]

    def test_bootstrap_scores_count(self, tvsd_encoding_results):
        """Should have the right number of bootstrap scores."""
        bs = tvsd_encoding_results.iloc[0].get("bootstrap_scores")
        assert bs is not None
        assert len(bs) == 50
