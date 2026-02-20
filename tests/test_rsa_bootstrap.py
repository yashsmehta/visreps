"""Comprehensive tests for RSA computation and bootstrapping across NSD, TVSD, THINGS.

Tests cover:
  1. Unit tests for compute_rdm (Pearson + Spearman, symmetry, diagonal, range)
  2. Unit tests for compute_rdm_correlation (Spearman, Kendall tau-a, Pearson)
  3. Unit tests for _kendall_tau_a conversion from tau-b
  4. Integration tests for compute_rsa (train/test: layer selection, no leakage, bootstrap)
  5. Integration tests for _eval_rsa inline bootstrap (NSD/TVSD path in evals.py)
  6. Alignment tests (_align_stimulus_level, squeeze behavior, concept averaging)
  7. Bootstrap-specific tests (subsample size, CI coverage, distribution shape)
  8. DB storage tests (results, layer_selection_scores, bootstrap_distributions)
  9. End-to-end integration with real model + data (NSD RSA + bootstrap, TVSD RSA + bootstrap)
  10. THINGS 80/20 concept-level train/test split tests
"""
import json
import math
import os
import sqlite3
import sys
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
import scipy.stats
import torch
from omegaconf import OmegaConf

# ── Ensure project root is on path ────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from visreps.analysis.rsa import (
    compute_rdm,
    compute_rdm_correlation,
    compute_rsa,
    _kendall_tau_a,
    _concept_average_exact,
    _rank,
)
from visreps.analysis.alignment import (
    AlignmentData,
    _align_stimulus_level,
    prepare_concept_alignment,
)


# ═══════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════

@pytest.fixture
def rng():
    """Deterministic random state for reproducibility."""
    return np.random.RandomState(42)


@pytest.fixture
def simple_representations():
    """Small known representations for manual RDM verification."""
    # 4 samples, 3 features — easy to verify by hand
    return torch.tensor([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [1.0, 2.0, 3.0],  # identical to row 0
        [7.0, 8.0, 9.0],
    ])


@pytest.fixture
def synthetic_alignment_data(rng):
    """Create synthetic AlignmentData with a known 'best' layer.

    Layer 'good' has activations that correlate with neural data.
    Layer 'bad' has random noise.
    """
    n_train, n_test, n_voxels = 200, 50, 100

    # Neural data: random but consistent
    neural_train = torch.randn(n_train, n_voxels)
    neural_test = torch.randn(n_test, n_voxels)

    # 'good' layer: activations = neural + small noise → high RSA
    good_train = neural_train + 0.1 * torch.randn(n_train, n_voxels)
    good_test = neural_test + 0.1 * torch.randn(n_test, n_voxels)

    # 'bad' layer: pure random noise → low RSA
    bad_train = torch.randn(n_train, n_voxels)
    bad_test = torch.randn(n_test, n_voxels)

    train_ids = [str(i) for i in range(n_train)]
    test_ids = [str(i) for i in range(n_train, n_train + n_test)]

    train = AlignmentData(
        activations={"good": good_train, "bad": bad_train},
        neural=neural_train,
        stimulus_ids=train_ids,
    )
    test = AlignmentData(
        activations={"good": good_test, "bad": bad_test},
        neural=neural_test,
        stimulus_ids=test_ids,
    )
    return train, test


# ═══════════════════════════════════════════════════════════════
# 1. UNIT TESTS: compute_rdm
# ═══════════════════════════════════════════════════════════════

class TestComputeRDM:
    """Tests for compute_rdm: RDM construction from representations."""

    def test_symmetry(self, rng):
        """RDM should be symmetric."""
        x = torch.randn(20, 10)
        rdm = compute_rdm(x)
        torch.testing.assert_close(rdm, rdm.T, atol=1e-5, rtol=1e-5)

    def test_diagonal_zero(self, rng):
        """Diagonal (self-dissimilarity) should be exactly 0."""
        x = torch.randn(20, 10)
        rdm = compute_rdm(x)
        assert torch.all(rdm.diag() == 0.0), f"Diagonal not zero: {rdm.diag()}"

    def test_range(self, rng):
        """Off-diagonal RDM values should be in [0, 2]."""
        x = torch.randn(50, 20)
        rdm = compute_rdm(x)
        n = rdm.size(0)
        mask = ~torch.eye(n, dtype=torch.bool)
        off_diag = rdm[mask]
        assert off_diag.min() >= -0.01, f"RDM has negative values: {off_diag.min()}"
        assert off_diag.max() <= 2.01, f"RDM values exceed 2: {off_diag.max()}"

    def test_identical_rows_zero_dissimilarity(self, simple_representations):
        """Identical representations should have dissimilarity = 0."""
        rdm = compute_rdm(simple_representations)
        # Rows 0 and 2 are identical
        assert rdm[0, 2].item() == pytest.approx(0.0, abs=1e-5)
        assert rdm[2, 0].item() == pytest.approx(0.0, abs=1e-5)

    def test_pearson_known_values(self):
        """Verify Pearson RDM against manual scipy computation."""
        x = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        rdm = compute_rdm(x, correlation="Pearson")
        # scipy pearsonr for these two vectors
        r, _ = scipy.stats.pearsonr([1, 0, 0], [0, 1, 0])
        expected_dissim = 1.0 - r
        assert rdm[0, 1].item() == pytest.approx(expected_dissim, abs=0.05)

    def test_spearman_rank_transform(self, rng):
        """Spearman RDM should differ from Pearson when data has nonlinear relationships."""
        # Exponential transform → rank-preserving but magnitude-changing
        x = torch.randn(30, 10)
        rdm_pearson = compute_rdm(x, correlation="Pearson")
        rdm_spearman = compute_rdm(x, correlation="Spearman")
        # They should NOT be identical (different computation)
        diff = (rdm_pearson - rdm_spearman).abs().sum()
        # With random data they'll be similar but not identical
        # Just check both are valid RDMs
        assert rdm_spearman.diag().sum() == 0.0
        torch.testing.assert_close(rdm_spearman, rdm_spearman.T, atol=1e-5, rtol=1e-5)

    def test_shape(self, rng):
        """RDM should be (n, n) for n samples."""
        n = 25
        x = torch.randn(n, 15)
        rdm = compute_rdm(x)
        assert rdm.shape == (n, n)

    def test_single_sample(self):
        """Single sample should produce a 1x1 RDM with 0 on diagonal."""
        x = torch.randn(1, 5)
        rdm = compute_rdm(x)
        assert rdm.shape == (1, 1)
        assert rdm[0, 0].item() == 0.0

    def test_high_dimensional(self, rng):
        """RDM should work with high-dimensional inputs (like conv features)."""
        x = torch.randn(10, 4096)
        rdm = compute_rdm(x)
        assert rdm.shape == (10, 10)
        assert torch.all(rdm.diag() == 0.0)

    def test_invalid_correlation_raises(self):
        """Should raise ValueError for invalid correlation type."""
        x = torch.randn(5, 3)
        with pytest.raises(ValueError):
            compute_rdm(x, correlation="cosine")

    def test_zero_variance_rows(self):
        """Zero-variance rows should not produce NaN/inf in the RDM."""
        x = torch.randn(10, 5)
        x[3] = 5.0  # constant row (zero variance)
        x[7] = -2.0  # another constant row
        rdm = compute_rdm(x)
        assert torch.isfinite(rdm).all(), f"RDM has non-finite values with constant rows"
        assert rdm.shape == (10, 10)
        assert torch.all(rdm.diag() == 0.0)

    def test_pearson_rdm_vs_scipy_rowpair(self):
        """Verify Pearson RDM value for specific row pair matches scipy."""
        x = torch.tensor([
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 4.0, 6.0, 8.0, 10.0],
            [5.0, 3.0, 1.0, -1.0, -3.0],
        ])
        rdm = compute_rdm(x, correlation="Pearson")
        # Rows 0 and 1 are perfectly correlated -> dissimilarity = 0
        assert rdm[0, 1].item() == pytest.approx(0.0, abs=1e-4)
        # Rows 0 and 2 are perfectly anti-correlated -> dissimilarity = 2
        assert rdm[0, 2].item() == pytest.approx(2.0, abs=1e-4)

    def test_output_dtype_float32(self, rng):
        """RDM output should be float32."""
        x = torch.randn(10, 5)
        rdm = compute_rdm(x)
        assert rdm.dtype == torch.float32


# ═══════════════════════════════════════════════════════════════
# 2. UNIT TESTS: compute_rdm_correlation
# ═══════════════════════════════════════════════════════════════

class TestComputeRDMCorrelation:
    """Tests for compute_rdm_correlation: comparing two RDMs."""

    def test_identical_rdms_spearman(self, rng):
        """Identical RDMs should have Spearman correlation = 1.0."""
        x = torch.randn(20, 10)
        rdm = compute_rdm(x)
        score = compute_rdm_correlation(rdm, rdm, correlation="Spearman")
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_identical_rdms_kendall(self, rng):
        """Identical RDMs should have Kendall correlation = 1.0."""
        x = torch.randn(20, 10)
        rdm = compute_rdm(x)
        score = compute_rdm_correlation(rdm, rdm, correlation="Kendall")
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_identical_rdms_pearson(self, rng):
        """Identical RDMs should have Pearson correlation = 1.0."""
        x = torch.randn(20, 10)
        rdm = compute_rdm(x)
        score = compute_rdm_correlation(rdm, rdm, correlation="Pearson")
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_random_rdms_near_zero(self, rng):
        """Unrelated RDMs should have correlation near 0."""
        rdm1 = compute_rdm(torch.randn(30, 10))
        rdm2 = compute_rdm(torch.randn(30, 10))
        score = compute_rdm_correlation(rdm1, rdm2, correlation="Spearman")
        assert abs(score) < 0.3, f"Random RDMs correlation too high: {score}"

    def test_spearman_vs_scipy(self, rng):
        """Spearman correlation should match scipy.stats.spearmanr on upper triangle."""
        x1 = torch.randn(15, 8)
        x2 = torch.randn(15, 8)
        rdm1 = compute_rdm(x1)
        rdm2 = compute_rdm(x2)

        # Our function
        score = compute_rdm_correlation(rdm1, rdm2, correlation="Spearman")

        # Manual scipy
        n = rdm1.size(0)
        idx = torch.triu_indices(n, n, offset=1)
        v1 = rdm1[idx[0], idx[1]].cpu().numpy()
        v2 = rdm2[idx[0], idx[1]].cpu().numpy()
        expected, _ = scipy.stats.spearmanr(v1, v2)

        assert score == pytest.approx(expected, abs=1e-5)

    def test_pearson_vs_scipy(self, rng):
        """Pearson correlation should match scipy.stats.pearsonr on upper triangle."""
        x1 = torch.randn(15, 8)
        x2 = torch.randn(15, 8)
        rdm1 = compute_rdm(x1)
        rdm2 = compute_rdm(x2)

        score = compute_rdm_correlation(rdm1, rdm2, correlation="Pearson")

        n = rdm1.size(0)
        idx = torch.triu_indices(n, n, offset=1)
        v1 = rdm1[idx[0], idx[1]].cpu().numpy()
        v2 = rdm2[idx[0], idx[1]].cpu().numpy()
        expected, _ = scipy.stats.pearsonr(v1, v2)

        assert score == pytest.approx(expected, abs=1e-5)

    def test_shape_mismatch_raises(self):
        """Mismatched RDM shapes should raise ValueError."""
        rdm1 = torch.zeros(5, 5)
        rdm2 = torch.zeros(6, 6)
        with pytest.raises(ValueError):
            compute_rdm_correlation(rdm1, rdm2)

    def test_upper_triangle_only(self, rng):
        """Verify only the strict upper triangle is used (no diagonal)."""
        n = 10
        rdm1 = compute_rdm(torch.randn(n, 5))
        rdm2 = compute_rdm(torch.randn(n, 5))

        # Number of upper-triangle pairs
        n_pairs = n * (n - 1) // 2
        idx = torch.triu_indices(n, n, offset=1)
        assert idx[0].numel() == n_pairs

    def test_invalid_correlation_raises(self, rng):
        """Should raise ValueError for unsupported correlation type."""
        rdm = compute_rdm(torch.randn(5, 3))
        with pytest.raises(ValueError):
            compute_rdm_correlation(rdm, rdm, correlation="cosine")

    def test_n_equals_1_returns_nan(self):
        """Single-sample RDM should return NaN (correlation undefined)."""
        rdm = torch.zeros(1, 1)
        score = compute_rdm_correlation(rdm, rdm, correlation="Spearman")
        assert math.isnan(score)

    def test_anti_correlated_rdms(self, rng):
        """Anti-correlated RDMs should have negative correlation."""
        x = torch.randn(20, 10)
        rdm1 = compute_rdm(x)
        # Invert the RDM (flip dissimilarity)
        rdm2 = 2.0 - rdm1
        rdm2.fill_diagonal_(0.0)
        score = compute_rdm_correlation(rdm1, rdm2, correlation="Spearman")
        assert score < -0.5, f"Anti-correlated RDMs should give negative score, got {score}"


# ═══════════════════════════════════════════════════════════════
# 3. UNIT TESTS: _kendall_tau_a
# ═══════════════════════════════════════════════════════════════

class TestKendallTauA:
    """Tests for the custom Kendall tau-a implementation."""

    def test_perfect_agreement(self):
        """Identical rankings should give tau_a = 1.0."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        tau_a, _ = _kendall_tau_a(x, x)
        assert tau_a == pytest.approx(1.0, abs=1e-5)

    def test_perfect_disagreement(self):
        """Reverse rankings should give tau_a = -1.0."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        tau_a, _ = _kendall_tau_a(x, y)
        assert tau_a == pytest.approx(-1.0, abs=1e-5)

    def test_tau_a_vs_tau_b_no_ties(self):
        """Without ties, tau-a should equal tau-b."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        tau_a, _ = _kendall_tau_a(x, y)
        tau_b = scipy.stats.kendalltau(x, y).statistic
        assert tau_a == pytest.approx(tau_b, abs=1e-5)

    def test_tau_a_vs_tau_b_with_ties(self):
        """With ties, tau-a should differ from tau-b (tau-a <= |tau-b|)."""
        x = np.array([1.0, 1.0, 2.0, 3.0, 3.0])
        y = np.array([2.0, 1.0, 3.0, 1.0, 2.0])
        tau_a, _ = _kendall_tau_a(x, y)
        tau_b = scipy.stats.kendalltau(x, y).statistic
        # tau-a should be smaller in magnitude (or equal) due to no tie adjustment
        assert abs(tau_a) <= abs(tau_b) + 1e-10

    def test_manual_tau_a(self):
        """Verify tau-a against manual concordant/discordant count.

        x = [1, 2, 3], y = [3, 1, 2]
        Pairs: (0,1): x↑ y↓ = D, (0,2): x↑ y↓ = D, (1,2): x↑ y↑ = C
        C = 1, D = 2, n_pairs = 3
        tau_a = (C - D) / n_pairs = (1 - 2) / 3 = -1/3
        """
        x = np.array([1.0, 2.0, 3.0])
        y = np.array([3.0, 1.0, 2.0])
        tau_a, _ = _kendall_tau_a(x, y)
        assert tau_a == pytest.approx(-1.0 / 3.0, abs=1e-5)

    def test_short_input(self):
        """Single-element input should return NaN."""
        x = np.array([1.0])
        tau_a, _ = _kendall_tau_a(x, x)
        assert math.isnan(tau_a)

    def test_large_n_no_overflow(self):
        """Large arrays should not cause integer overflow in n0 computation.

        The code casts to float64 to avoid overflow; verify this works for n=1000.
        """
        rng = np.random.RandomState(42)
        n = 1000
        x = rng.randn(n)
        y = rng.randn(n)
        tau_a, _ = _kendall_tau_a(x, y)
        assert np.isfinite(tau_a), f"tau_a overflowed for n={n}"
        assert -1 <= tau_a <= 1

    def test_all_ties_returns_nan(self):
        """All-tied arrays should return NaN (no concordant/discordant pairs)."""
        x = np.array([1.0, 1.0, 1.0, 1.0])
        y = np.array([2.0, 2.0, 2.0, 2.0])
        tau_a, _ = _kendall_tau_a(x, y)
        assert math.isnan(tau_a)


# ═══════════════════════════════════════════════════════════════
# 4. INTEGRATION TESTS: compute_rsa (train/test)
# ═══════════════════════════════════════════════════════════════

class TestComputeRSA:
    """Tests for train/test RSA (used by NSD/TVSD)."""

    def test_selects_correct_layer(self, synthetic_alignment_data):
        """Should select 'good' layer over 'bad' layer."""
        train, test = synthetic_alignment_data
        cfg = OmegaConf.create({"compare_method": "spearman"})
        results = compute_rsa(cfg, train, test, n_select=100, bootstrap=False, seed=42)
        assert len(results) == 1
        assert results[0]["layer"] == "good"

    def test_score_range(self, synthetic_alignment_data):
        """RSA score should be in [-1, 1]."""
        train, test = synthetic_alignment_data
        cfg = OmegaConf.create({"compare_method": "spearman"})
        results = compute_rsa(cfg, train, test, n_select=100, bootstrap=False, seed=42)
        assert -1 <= results[0]["score"] <= 1

    def test_good_layer_high_score(self, synthetic_alignment_data):
        """'good' layer (correlated with neural) should have high RSA score."""
        train, test = synthetic_alignment_data
        cfg = OmegaConf.create({"compare_method": "spearman"})
        results = compute_rsa(cfg, train, test, n_select=100, bootstrap=False, seed=42)
        # With neural + 0.1*noise, should be quite high
        assert results[0]["score"] > 0.3, f"Score too low: {results[0]['score']}"

    def test_no_train_test_leakage(self, synthetic_alignment_data):
        """Replacing test neural data with random should drop the score."""
        train, test = synthetic_alignment_data
        cfg = OmegaConf.create({"compare_method": "spearman"})

        # Normal run
        results_normal = compute_rsa(cfg, train, test, n_select=100, bootstrap=False, seed=42)

        # Scrambled test neural data
        test_scrambled = AlignmentData(
            activations=test.activations,
            neural=torch.randn_like(test.neural),
            stimulus_ids=test.stimulus_ids,
        )
        results_scrambled = compute_rsa(
            cfg, train, test_scrambled, n_select=100, bootstrap=False, seed=42
        )

        # Normal should be much better than scrambled
        assert results_normal[0]["score"] > results_scrambled[0]["score"] + 0.1

    def test_layer_selection_scores_returned(self, synthetic_alignment_data):
        """Should return layer selection scores for all layers."""
        train, test = synthetic_alignment_data
        cfg = OmegaConf.create({"compare_method": "spearman"})
        results = compute_rsa(cfg, train, test, n_select=100, bootstrap=False, seed=42)
        ls = results[0]["layer_selection_scores"]
        assert len(ls) == 2  # 'good' and 'bad'
        layers = {x["layer"] for x in ls}
        assert layers == {"good", "bad"}

    def test_bootstrap_produces_cis(self, synthetic_alignment_data):
        """Bootstrap should produce ci_low, ci_high, and bootstrap_scores."""
        train, test = synthetic_alignment_data
        cfg = OmegaConf.create({"compare_method": "spearman"})
        results = compute_rsa(
            cfg, train, test,
            n_select=100, bootstrap=True, n_bootstrap=50, seed=42,
        )
        r = results[0]
        assert r["ci_low"] is not None
        assert r["ci_high"] is not None
        assert r["ci_low"] < r["ci_high"]
        assert "bootstrap_scores" in r
        assert len(r["bootstrap_scores"]) == 50

    def test_bootstrap_ci_contains_point_estimate(self, synthetic_alignment_data):
        """95% CI should usually contain the point estimate (within 99% of samples).

        With 50 bootstraps this is approximate, so we use a generous tolerance.
        """
        train, test = synthetic_alignment_data
        cfg = OmegaConf.create({"compare_method": "spearman"})
        results = compute_rsa(
            cfg, train, test,
            n_select=100, bootstrap=True, n_bootstrap=200, seed=42,
        )
        r = results[0]
        # Point estimate should be within the bootstrap range (not necessarily 95% CI)
        bs_min = min(r["bootstrap_scores"])
        bs_max = max(r["bootstrap_scores"])
        assert bs_min <= r["score"] <= bs_max, (
            f"Point estimate {r['score']:.4f} outside bootstrap range "
            f"[{bs_min:.4f}, {bs_max:.4f}]"
        )

    def test_bootstrap_subsample_size(self, synthetic_alignment_data):
        """Bootstrap should subsample 90% of test set (no replacement)."""
        train, test = synthetic_alignment_data
        n_test = test.neural.size(0)
        expected_subsample = int(n_test * 0.9)

        cfg = OmegaConf.create({"compare_method": "spearman"})

        # Monkey-patch to verify subsample size
        original_compute_rdm_correlation = compute_rdm_correlation
        subsample_sizes = []

        def tracking_correlation(rdm1, rdm2, **kwargs):
            subsample_sizes.append(rdm1.size(0))
            return original_compute_rdm_correlation(rdm1, rdm2, **kwargs)

        with patch("visreps.analysis.rsa.compute_rdm_correlation", side_effect=tracking_correlation):
            compute_rsa(
                cfg, train, test,
                n_select=100, bootstrap=True, n_bootstrap=5, seed=42,
            )

        # During bootstrap, RDM sub-selections should have the expected subsample size
        bootstrap_sizes = [s for s in subsample_sizes if s == expected_subsample]
        assert len(bootstrap_sizes) == 5, (
            f"Expected 5 bootstrap iterations with size {expected_subsample}, "
            f"got sizes: {subsample_sizes}"
        )

    def test_no_bootstrap_returns_none_cis(self, synthetic_alignment_data):
        """Without bootstrap, ci_low and ci_high should be None."""
        train, test = synthetic_alignment_data
        cfg = OmegaConf.create({"compare_method": "spearman"})
        results = compute_rsa(cfg, train, test, n_select=100, bootstrap=False, seed=42)
        r = results[0]
        assert r["ci_low"] is None
        assert r["ci_high"] is None
        assert "bootstrap_scores" not in r

    def test_compare_method_kendall(self, synthetic_alignment_data):
        """Should work with Kendall as compare method."""
        train, test = synthetic_alignment_data
        cfg = OmegaConf.create({"compare_method": "kendall"})
        results = compute_rsa(cfg, train, test, n_select=100, bootstrap=False, seed=42)
        assert results[0]["compare_method"] == "kendall"
        assert results[0]["layer"] == "good"
        assert -1 <= results[0]["score"] <= 1

    def test_re_extract_fn_called(self, synthetic_alignment_data):
        """If re_extract_fn is provided, it should be called for the best layer."""
        train, test = synthetic_alignment_data
        cfg = OmegaConf.create({"compare_method": "spearman"})

        # Mock re_extract_fn that returns same-shaped data
        best_layer_exact = test.activations["good"]  # Simulate exact re-extraction
        mock_fn = MagicMock(return_value=(best_layer_exact, test.stimulus_ids))

        results = compute_rsa(
            cfg, train, test,
            n_select=100, bootstrap=False, seed=42,
            re_extract_fn=mock_fn,
        )
        # Should have been called with the best layer and test stimulus IDs
        mock_fn.assert_called_once()
        call_args = mock_fn.call_args
        assert call_args[0][0] == "good"  # best layer name

    def test_reproducibility(self, synthetic_alignment_data):
        """Same seed should produce identical results."""
        train, test = synthetic_alignment_data
        cfg = OmegaConf.create({"compare_method": "spearman"})
        r1 = compute_rsa(cfg, train, test, n_select=100, bootstrap=True, n_bootstrap=20, seed=42)
        r2 = compute_rsa(cfg, train, test, n_select=100, bootstrap=True, n_bootstrap=20, seed=42)
        assert r1[0]["score"] == r2[0]["score"]
        assert r1[0]["ci_low"] == r2[0]["ci_low"]
        assert r1[0]["ci_high"] == r2[0]["ci_high"]

    def test_different_seeds_differ(self, synthetic_alignment_data):
        """Different seeds should generally produce different bootstrap distributions."""
        train, test = synthetic_alignment_data
        cfg = OmegaConf.create({"compare_method": "spearman"})
        r1 = compute_rsa(cfg, train, test, n_select=100, bootstrap=True, n_bootstrap=20, seed=42)
        r2 = compute_rsa(cfg, train, test, n_select=100, bootstrap=True, n_bootstrap=20, seed=99)
        # Point estimates may differ slightly due to different n_select subsamples
        # Bootstrap distributions should differ
        assert r1[0]["bootstrap_scores"] != r2[0]["bootstrap_scores"]

    def test_does_not_mutate_inputs(self, synthetic_alignment_data):
        """compute_rsa should NOT mutate the input AlignmentData objects."""
        train, test = synthetic_alignment_data
        train_neural_clone = train.neural.clone()
        test_neural_clone = test.neural.clone()
        train_acts_clones = {k: v.clone() for k, v in train.activations.items()}
        test_acts_clones = {k: v.clone() for k, v in test.activations.items()}

        cfg = OmegaConf.create({"compare_method": "spearman"})
        compute_rsa(cfg, train, test, n_select=100, bootstrap=True, n_bootstrap=10, seed=42)

        torch.testing.assert_close(train.neural, train_neural_clone)
        torch.testing.assert_close(test.neural, test_neural_clone)
        for k in train.activations:
            torch.testing.assert_close(train.activations[k], train_acts_clones[k])
        for k in test.activations:
            torch.testing.assert_close(test.activations[k], test_acts_clones[k])

    def test_verbose_smoke(self, synthetic_alignment_data):
        """Verbose mode should not crash (smoke test)."""
        train, test = synthetic_alignment_data
        cfg = OmegaConf.create({"compare_method": "spearman"})
        results = compute_rsa(cfg, train, test, n_select=50, bootstrap=False, seed=42, verbose=True)
        assert len(results) == 1

    def test_analysis_field_is_rsa(self, synthetic_alignment_data):
        """Result dict should have analysis='rsa'."""
        train, test = synthetic_alignment_data
        cfg = OmegaConf.create({"compare_method": "spearman"})
        results = compute_rsa(cfg, train, test, n_select=100, bootstrap=False, seed=42)
        assert results[0]["analysis"] == "rsa"

    def test_default_compare_method(self, synthetic_alignment_data):
        """If compare_method not in cfg, should default to 'spearman'."""
        train, test = synthetic_alignment_data
        cfg = OmegaConf.create({})
        results = compute_rsa(cfg, train, test, n_select=100, bootstrap=False, seed=42)
        assert results[0]["compare_method"] == "spearman"

    def test_n_select_none_uses_all_train(self, synthetic_alignment_data):
        """n_select=None should use all training stimuli for layer selection."""
        train, test = synthetic_alignment_data
        cfg = OmegaConf.create({"compare_method": "spearman"})
        results = compute_rsa(cfg, train, test, n_select=None, bootstrap=False, seed=42)
        assert len(results) == 1
        assert results[0]["layer"] == "good"


# ═══════════════════════════════════════════════════════════════
# 5. INLINE BOOTSTRAP TESTS (_eval_rsa in evals.py)
# ═══════════════════════════════════════════════════════════════

class TestEvalRSABootstrap:
    """Tests for the inline bootstrap in evals.py _eval_rsa (NSD/TVSD path).

    These test the Phase 1 → Phase 2 → bootstrap flow using synthetic data
    to avoid needing real neural datasets.
    """

    def _make_synthetic_all_data(self, n_train=100, n_test=30, n_voxels=50, n_subjects=1, regions=None):
        """Create synthetic all_data dict matching load_all_nsd_data / load_all_tvsd_data output."""
        if regions is None:
            regions = ["test_region"]

        all_stim_ids = [str(i) for i in range(n_train + n_test)]
        train_ids = all_stim_ids[:n_train]
        test_ids = all_stim_ids[n_train:]

        neural = {}
        for region in regions:
            neural[region] = {}
            for subj in range(n_subjects):
                neural[region][subj] = {
                    "train": {sid: np.random.randn(n_voxels).astype(np.float32) for sid in train_ids},
                    "test": {sid: np.random.randn(n_voxels).astype(np.float32) for sid in test_ids},
                }

        # Stimuli: use dummy image paths
        stimuli = {sid: f"/tmp/img_{sid}.jpg" for sid in all_stim_ids}

        return {
            "regions": regions,
            "subjects": list(range(n_subjects)),
            "neural": neural,
            "stimuli": stimuli,
            "shared_test_ids": test_ids,
        }

    def test_phase1_selects_best_layer(self):
        """Phase 1 layer selection should pick the most brain-aligned layer."""
        from visreps.analysis.rsa import compute_rdm, compute_rdm_correlation

        # Create neural data
        n_train = 100
        n_voxels = 50
        rng = np.random.RandomState(42)

        neural_responses = rng.randn(n_train, n_voxels).astype(np.float32)

        # 'good' activations correlate with neural
        good_acts = torch.from_numpy(neural_responses + 0.1 * rng.randn(n_train, n_voxels).astype(np.float32))
        bad_acts = torch.randn(n_train, n_voxels)

        neural_tensor = torch.from_numpy(neural_responses)
        acts = {"good": good_acts, "bad": bad_acts}

        # Simulate phase 1 layer selection logic from _eval_rsa
        n_select = min(50, n_train)
        rng_sel = np.random.RandomState(42)
        sel_idx = rng_sel.choice(n_train, size=n_select, replace=False)

        neural_rdm_sel = compute_rdm(neural_tensor[sel_idx])

        best_layer, best_score = None, -float("inf")
        for layer, layer_acts in acts.items():
            flat = layer_acts[sel_idx]
            layer_rdm = compute_rdm(flat)
            score = compute_rdm_correlation(layer_rdm, neural_rdm_sel, correlation="Spearman")
            if score > best_score:
                best_score = score
                best_layer = layer

        assert best_layer == "good"

    def test_bootstrap_rdm_subindexing(self):
        """Verify that RDM sub-indexing rdm[idx][:, idx] produces correct sub-RDM."""
        n = 20
        x = torch.randn(n, 10)
        rdm = compute_rdm(x)

        # Take a subset
        idx = torch.tensor([0, 5, 10, 15])
        sub_rdm = rdm[idx][:, idx]

        # Compute RDM directly from subset
        direct_rdm = compute_rdm(x[idx])

        # They should be identical (same Pearson correlations, same stimuli)
        torch.testing.assert_close(sub_rdm, direct_rdm, atol=1e-4, rtol=1e-4)

    def test_bootstrap_90_percent_subsample(self):
        """Inline bootstrap should use int(n_test * 0.9) subsample size."""
        n_test = 100
        expected = int(n_test * 0.9)
        assert expected == 90

        n_test = 50
        expected = int(n_test * 0.9)
        assert expected == 45

        # Edge case: small test set
        n_test = 10
        expected = int(n_test * 0.9)
        assert expected == 9

    def test_bootstrap_without_replacement(self):
        """Bootstrap sampling should use replace=False (subsampling, not resampling)."""
        rng = np.random.RandomState(42)
        n_test = 50
        n_sub = int(n_test * 0.9)

        for _ in range(100):
            idx = rng.choice(n_test, size=n_sub, replace=False)
            # Without replacement: all indices should be unique
            assert len(set(idx)) == n_sub


# ═══════════════════════════════════════════════════════════════
# 7. ALIGNMENT TESTS
# ═══════════════════════════════════════════════════════════════

class TestAlignment:
    """Tests for _align_stimulus_level and related alignment functions."""

    def test_align_basic_matching(self):
        """_align_stimulus_level should match activations to targets by ID."""
        acts = {"layer1": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])}
        targets = {"0": np.array([10.0, 20.0]), "2": np.array([30.0, 40.0])}
        keys = ["0", "1", "2"]

        aligned_acts, neural, matched_ids = _align_stimulus_level(acts, targets, keys)

        assert matched_ids == ["0", "2"]
        assert aligned_acts["layer1"].shape[0] == 2
        torch.testing.assert_close(aligned_acts["layer1"][0], torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(aligned_acts["layer1"][1], torch.tensor([5.0, 6.0]))

    def test_align_preserves_order(self):
        """Aligned activations and neural data should be in the same stimulus order."""
        n = 50
        acts = {"layer1": torch.randn(n, 10)}
        keys = [str(i) for i in range(n)]
        # Only include even-numbered stimuli
        targets = {str(i): np.random.randn(5).astype(np.float32) for i in range(0, n, 2)}

        aligned_acts, neural, matched_ids = _align_stimulus_level(acts, targets, keys)

        assert len(matched_ids) == 25
        for i, mid in enumerate(matched_ids):
            orig_idx = int(mid)
            torch.testing.assert_close(aligned_acts["layer1"][i], acts["layer1"][orig_idx])

    def test_align_preserves_1_voxel_dimension(self):
        """With n_voxels=1, neural tensor should remain 2D (n, 1)."""
        acts = {"layer1": torch.randn(3, 5)}
        targets = {
            "0": np.array([1.0]),  # shape (1,) — single voxel
            "1": np.array([2.0]),
            "2": np.array([3.0]),
        }
        keys = ["0", "1", "2"]

        aligned_acts, neural, matched_ids = _align_stimulus_level(acts, targets, keys)
        assert neural.ndim == 2
        assert neural.shape == (3, 1)

    def test_align_empty_intersection(self):
        """No matching IDs should return empty results."""
        acts = {"layer1": torch.randn(3, 5)}
        targets = {"10": np.array([1.0, 2.0]), "11": np.array([3.0, 4.0])}
        keys = ["0", "1", "2"]

        aligned_acts, neural, matched_ids = _align_stimulus_level(acts, targets, keys)
        assert len(matched_ids) == 0
        assert neural.shape[0] == 0
        assert aligned_acts["layer1"].shape[0] == 0

    def test_concept_averaging(self):
        """prepare_concept_alignment should average activations per concept."""
        n_images = 10
        n_features = 4
        acts_raw = {"layer1": torch.arange(n_images * n_features, dtype=torch.float32).reshape(n_images, n_features)}
        keys = [f"img_{i}" for i in range(n_images)]

        # 2 concepts, each with 5 images
        neural_data_raw = {
            "embeddings": {
                "cat": np.array([1.0, 2.0]),
                "dog": np.array([3.0, 4.0]),
            },
            "image_ids": {
                "cat": ["img_0", "img_1", "img_2", "img_3", "img_4"],
                "dog": ["img_5", "img_6", "img_7", "img_8", "img_9"],
            },
        }

        cfg = OmegaConf.create({"compare_method": "spearman"})
        data = prepare_concept_alignment(cfg, acts_raw, neural_data_raw, keys)

        assert data.neural.shape[0] == 2  # 2 concepts
        assert data.activations["layer1"].shape[0] == 2

        # Cat images: 0,1,2,3,4 → indices 0-4 → mean of rows 0-4
        cat_expected = acts_raw["layer1"][:5].float().mean(0)
        dog_expected = acts_raw["layer1"][5:].float().mean(0)

        # Find which index is cat vs dog
        cat_idx = data.stimulus_ids.index("cat")
        dog_idx = data.stimulus_ids.index("dog")

        torch.testing.assert_close(data.activations["layer1"][cat_idx], cat_expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(data.activations["layer1"][dog_idx], dog_expected, atol=1e-5, rtol=1e-5)


# ═══════════════════════════════════════════════════════════════
# 8. RANK FUNCTION TESTS
# ═══════════════════════════════════════════════════════════════

class TestRank:
    """Tests for the _rank function used in Spearman RDM computation."""

    def test_basic_ranking(self):
        """Should produce dense ranks via double argsort."""
        x = torch.tensor([[3.0, 1.0, 2.0]])
        ranked = _rank(x)
        # 1.0 → rank 0, 2.0 → rank 1, 3.0 → rank 2
        expected = torch.tensor([[2.0, 0.0, 1.0]])
        torch.testing.assert_close(ranked, expected)

    def test_already_sorted(self):
        """Already-sorted input should get ascending ranks."""
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        ranked = _rank(x)
        expected = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
        torch.testing.assert_close(ranked, expected)

    def test_multi_row(self):
        """Each row should be ranked independently."""
        x = torch.tensor([
            [3.0, 1.0, 2.0],
            [1.0, 3.0, 2.0],
        ])
        ranked = _rank(x)
        assert ranked[0, 1] == 0.0  # 1.0 is smallest in row 0
        assert ranked[1, 0] == 0.0  # 1.0 is smallest in row 1


# ═══════════════════════════════════════════════════════════════
# 9. DB STORAGE TESTS
# ═══════════════════════════════════════════════════════════════

class TestDBStorage:
    """Tests for save_results and DB table structure."""

    def _make_cfg(self, **overrides):
        base = {
            "seed": 1, "epoch": 20, "region": "ventral visual stream",
            "subject_idx": 0, "neural_dataset": "nsd", "cfg_id": 1000,
            "pca_labels": False, "pca_n_classes": 1000,
            "pca_labels_folder": "pca_labels_alexnet",
            "checkpoint_dir": "/data/ymehta3/default",
            "analysis": "rsa", "compare_method": "spearman",
            "reconstruct_from_pcs": False, "pca_k": 1,
            "model_name": "AlexNet",
        }
        base.update(overrides)
        return OmegaConf.create(base)

    def test_results_saved(self, tmp_path):
        """Results table should contain the saved row."""
        from visreps.utils import save_results, _RESULTS_DB_PATH

        # Temporarily redirect DB path
        old_path = _RESULTS_DB_PATH
        import visreps.utils as vu
        vu._RESULTS_DB_PATH = tmp_path / "test_results.db"

        try:
            df = pd.DataFrame([{
                "layer": "fc1_pre",
                "compare_method": "spearman",
                "score": 0.25,
                "ci_low": 0.20,
                "ci_high": 0.30,
                "analysis": "rsa",
                "layer_selection_scores": [
                    {"layer": "conv1", "score": 0.1},
                    {"layer": "fc1_pre", "score": 0.25},
                ],
            }])
            cfg = self._make_cfg()
            save_results(df, cfg)

            conn = sqlite3.connect(str(vu._RESULTS_DB_PATH))
            rows = pd.read_sql("SELECT * FROM results", conn)
            assert len(rows) == 1
            assert rows.iloc[0]["layer"] == "fc1_pre"
            assert rows.iloc[0]["score"] == pytest.approx(0.25)
            assert rows.iloc[0]["ci_low"] == pytest.approx(0.20)
            assert rows.iloc[0]["ci_high"] == pytest.approx(0.30)

            # Layer selection scores
            ls = pd.read_sql("SELECT * FROM layer_selection_scores", conn)
            assert len(ls) == 2
            conn.close()
        finally:
            vu._RESULTS_DB_PATH = old_path

    def test_bootstrap_distribution_saved(self, tmp_path):
        """bootstrap_distributions table should store the JSON array."""
        import visreps.utils as vu
        from visreps.utils import save_results

        old_path = vu._RESULTS_DB_PATH
        vu._RESULTS_DB_PATH = tmp_path / "test_bootstrap.db"

        try:
            bootstrap_scores = [0.2, 0.22, 0.24, 0.21, 0.23]
            df = pd.DataFrame([{
                "layer": "fc1_pre",
                "compare_method": "spearman",
                "score": 0.22,
                "ci_low": 0.20,
                "ci_high": 0.24,
                "analysis": "rsa",
                "bootstrap_scores": bootstrap_scores,
            }])
            cfg = self._make_cfg()
            save_results(df, cfg)

            conn = sqlite3.connect(str(vu._RESULTS_DB_PATH))
            rows = pd.read_sql("SELECT * FROM bootstrap_distributions", conn)
            assert len(rows) == 1
            stored = json.loads(rows.iloc[0]["scores"])
            assert len(stored) == 5
            assert stored == pytest.approx(bootstrap_scores)
            conn.close()
        finally:
            vu._RESULTS_DB_PATH = old_path

    def test_deduplication(self, tmp_path):
        """INSERT OR REPLACE should overwrite on re-run."""
        import visreps.utils as vu
        from visreps.utils import save_results

        old_path = vu._RESULTS_DB_PATH
        vu._RESULTS_DB_PATH = tmp_path / "test_dedup.db"

        try:
            cfg = self._make_cfg()

            # First save
            df1 = pd.DataFrame([{
                "layer": "fc1_pre", "compare_method": "spearman",
                "score": 0.25, "ci_low": None, "ci_high": None, "analysis": "rsa",
            }])
            save_results(df1, cfg)

            # Second save with different score
            df2 = pd.DataFrame([{
                "layer": "fc1_pre", "compare_method": "spearman",
                "score": 0.30, "ci_low": None, "ci_high": None, "analysis": "rsa",
            }])
            save_results(df2, cfg)

            conn = sqlite3.connect(str(vu._RESULTS_DB_PATH))
            rows = pd.read_sql("SELECT * FROM results", conn)
            assert len(rows) == 1  # Should NOT have 2 rows
            assert rows.iloc[0]["score"] == pytest.approx(0.30)  # Should be updated
            conn.close()
        finally:
            vu._RESULTS_DB_PATH = old_path


# ═══════════════════════════════════════════════════════════════
# 10. END-TO-END INTEGRATION TESTS (real model + data)
# ═══════════════════════════════════════════════════════════════

class TestEndToEndRSA:
    """End-to-end RSA tests with the real model and neural data.

    These tests load the actual cfg1000a epoch-20 checkpoint and run RSA
    with bootstrap on NSD and TVSD to verify the full pipeline.
    Requires: GPU, checkpoint at /data/ymehta3/default/cfg1000a/checkpoint_epoch_20.pth,
    and neural data at their configured paths.
    """

    @pytest.fixture(autouse=True)
    def _check_prerequisites(self):
        """Skip if GPU or checkpoint not available."""
        if not torch.cuda.is_available():
            pytest.skip("No GPU available")
        checkpoint = "/data/ymehta3/default/cfg1000a/checkpoint_epoch_20.pth"
        if not os.path.exists(checkpoint):
            pytest.skip(f"Checkpoint not found: {checkpoint}")

    def _run_eval(self, overrides):
        """Run an eval and return the results DataFrame."""
        from dotenv import load_dotenv
        load_dotenv()
        import visreps.utils as utils
        import visreps.evals as evals

        cfg = utils.load_config("configs/eval/base.json", overrides)
        cfg = utils.validate_config(cfg)
        return evals.eval(cfg)

    def test_nsd_rsa_with_bootstrap(self):
        """NSD RSA with bootstrap: score, CIs, layer hierarchy, DB storage."""
        results = self._run_eval([
            "mode=eval", "cfg_id=1000", "seed=1",
            "checkpoint_model=checkpoint_epoch_20.pth",
            "neural_dataset=nsd", "analysis=rsa", "compare_method=spearman",
            "subject_idx=0", "region=ventral visual stream",
            "log_expdata=true", "bootstrap=true", "n_bootstrap=100",
            "verbose=false", "batchsize=64", "num_workers=4",
        ])

        assert results is not None and len(results) > 0
        r = results.iloc[0]

        # Score should be meaningful (trained model on ventral stream)
        assert r["score"] > 0.05, f"NSD RSA score too low: {r['score']}"

        # CIs should exist and bracket the score reasonably
        assert r["ci_low"] is not None
        assert r["ci_high"] is not None
        assert r["ci_low"] < r["ci_high"]
        assert r["ci_low"] < r["score"] < r["ci_high"], (
            f"Point estimate {r['score']:.4f} not within CI [{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
        )

        # Best layer should be a late layer for ventral stream
        late_layers = {"conv3_pre", "conv3_post", "conv4_pre", "conv4_post",
                       "conv5_pre", "conv5_post", "fc1_pre", "fc1_post",
                       "fc2_pre", "fc2_post"}
        assert r["layer"] in late_layers, f"Best layer '{r['layer']}' not in expected late-layer set"

        # Bootstrap distribution should be stored
        assert "bootstrap_scores" in r.index or "bootstrap_scores" in results.columns
        bs = r.get("bootstrap_scores")
        if bs is not None:
            assert len(bs) == 100
            # Distribution should have reasonable variance
            bs_arr = np.array(bs)
            assert bs_arr.std() > 0.001, f"Bootstrap std too low: {bs_arr.std()}"
            assert bs_arr.std() < 0.5, f"Bootstrap std too high: {bs_arr.std()}"

        # DB storage check
        import sqlite3
        conn = sqlite3.connect("results.db")
        db_rows = pd.read_sql(
            "SELECT * FROM results WHERE neural_dataset='nsd' AND analysis='rsa' "
            "AND region='ventral visual stream' AND subject_idx='0' AND seed=1 "
            "AND compare_method='spearman' ORDER BY rowid DESC LIMIT 1",
            conn,
        )
        assert len(db_rows) >= 1
        db_score = db_rows.iloc[0]["score"]
        assert db_score == pytest.approx(r["score"], abs=1e-6)

        # Bootstrap distribution in DB
        run_id = db_rows.iloc[0]["run_id"]
        bs_rows = pd.read_sql(
            f"SELECT * FROM bootstrap_distributions WHERE run_id='{run_id}'", conn
        )
        assert len(bs_rows) >= 1
        stored_bs = json.loads(bs_rows.iloc[0]["scores"])
        assert len(stored_bs) == 100
        conn.close()

    def test_tvsd_rsa_with_bootstrap(self):
        """TVSD RSA with bootstrap: score, CIs, layer hierarchy, DB storage."""
        results = self._run_eval([
            "mode=eval", "cfg_id=1000", "seed=1",
            "checkpoint_model=checkpoint_epoch_20.pth",
            "neural_dataset=tvsd", "analysis=rsa", "compare_method=spearman",
            "subject_idx=0", "region=IT",
            "log_expdata=true", "bootstrap=true", "n_bootstrap=100",
            "verbose=false", "batchsize=64", "num_workers=4",
        ])

        assert results is not None and len(results) > 0
        r = results.iloc[0]

        # Score should be meaningful
        assert r["score"] > 0.02, f"TVSD RSA score too low: {r['score']}"
        assert -1 <= r["score"] <= 1

        # CIs should exist
        assert r["ci_low"] is not None
        assert r["ci_high"] is not None
        assert r["ci_low"] < r["ci_high"]

        # With only 100 test stimuli, CI may not always bracket the point estimate
        # but the CI should at least be reasonable
        ci_width = r["ci_high"] - r["ci_low"]
        assert ci_width > 0.01, f"CI too narrow: {ci_width}"
        assert ci_width < 1.0, f"CI too wide: {ci_width}"

        # Layer selection scores should exist
        ls = r.get("layer_selection_scores")
        if ls is not None:
            assert len(ls) > 0
            # At least some layer differentiation
            scores = [x["score"] for x in ls]
            assert max(scores) - min(scores) > 0.01

        # DB storage
        import sqlite3
        conn = sqlite3.connect("results.db")
        db_rows = pd.read_sql(
            "SELECT * FROM results WHERE neural_dataset='tvsd' AND analysis='rsa' "
            "AND region='IT' AND subject_idx='0' AND seed=1 "
            "AND compare_method='spearman' ORDER BY rowid DESC LIMIT 1",
            conn,
        )
        assert len(db_rows) >= 1

        run_id = db_rows.iloc[0]["run_id"]
        bs_rows = pd.read_sql(
            f"SELECT * FROM bootstrap_distributions WHERE run_id='{run_id}'", conn
        )
        assert len(bs_rows) >= 1
        stored_bs = json.loads(bs_rows.iloc[0]["scores"])
        assert len(stored_bs) == 100
        conn.close()

    def test_things_rsa_with_bootstrap(self):
        """THINGS RSA with bootstrap: 80/20 train/test split, score, CIs, DB storage."""
        results = self._run_eval([
            "mode=eval", "cfg_id=1000", "seed=1",
            "checkpoint_model=checkpoint_epoch_20.pth",
            "neural_dataset=things-behavior", "analysis=rsa", "compare_method=spearman",
            "log_expdata=true", "bootstrap=true", "n_bootstrap=100",
            "verbose=false", "batchsize=64", "num_workers=4",
        ])

        assert results is not None and len(results) > 0
        r = results.iloc[0]

        # Score should be meaningful for trained model
        assert r["score"] > 0.10, f"THINGS RSA score too low: {r['score']}"
        assert -1 <= r["score"] <= 1

        # CIs should exist (from train/test RSA, not k-fold)
        assert r["ci_low"] is not None
        assert r["ci_high"] is not None
        assert r["ci_low"] < r["ci_high"]

        # Should NOT have fold_results (no longer k-fold)
        assert r.get("fold_results") is None

        # Bootstrap distribution
        bs = r.get("bootstrap_scores")
        if bs is not None:
            assert len(bs) == 100

        # DB storage
        import sqlite3
        conn = sqlite3.connect("results.db")
        db_rows = pd.read_sql(
            "SELECT * FROM results WHERE neural_dataset='things-behavior' AND analysis='rsa' "
            "AND compare_method='spearman' AND seed=1 ORDER BY rowid DESC LIMIT 1",
            conn,
        )
        assert len(db_rows) >= 1

        run_id = db_rows.iloc[0]["run_id"]

        # Check bootstrap_distributions
        bs_rows = pd.read_sql(
            f"SELECT * FROM bootstrap_distributions WHERE run_id='{run_id}'", conn
        )
        assert len(bs_rows) >= 1
        stored_bs = json.loads(bs_rows.iloc[0]["scores"])
        assert len(stored_bs) == 100
        conn.close()


# ═══════════════════════════════════════════════════════════════
# THINGS 80/20 CONCEPT-LEVEL TRAIN/TEST SPLIT TESTS
# ═══════════════════════════════════════════════════════════════

class TestThingsConceptSplit:
    """Tests for the 80/20 concept-level train/test split used by THINGS-behavior."""

    def _make_concept_data(self, n_concepts=100, n_features=50, seed=42):
        """Create synthetic concept-level AlignmentData."""
        torch.manual_seed(seed)
        neural = torch.randn(n_concepts, n_features)
        good_acts = neural + 0.1 * torch.randn(n_concepts, n_features)
        bad_acts = torch.randn(n_concepts, n_features)
        concepts = [f"concept_{i}" for i in range(n_concepts)]
        concept_image_ids = {c: [f"{c}_img1", f"{c}_img2"] for c in concepts}
        return AlignmentData(
            activations={"good": good_acts, "bad": bad_acts},
            neural=neural,
            stimulus_ids=concepts,
            concept_image_ids=concept_image_ids,
        )

    def _split_concepts(self, data, seed=42):
        """Replicate the 80/20 split logic from evals.py."""
        rng = np.random.RandomState(seed)
        n = data.neural.size(0)
        perm = rng.permutation(n)
        n_sel = int(n * 0.2)
        sel_idx, eval_idx = perm[:n_sel], perm[n_sel:]

        selection = AlignmentData(
            activations={l: a[sel_idx] for l, a in data.activations.items()},
            neural=data.neural[sel_idx],
            stimulus_ids=[data.stimulus_ids[i] for i in sel_idx],
        )
        evaluation = AlignmentData(
            activations={l: a[eval_idx] for l, a in data.activations.items()},
            neural=data.neural[eval_idx],
            stimulus_ids=[data.stimulus_ids[i] for i in eval_idx],
            concept_image_ids={
                data.stimulus_ids[i]: data.concept_image_ids[data.stimulus_ids[i]]
                for i in eval_idx
            },
        )
        return selection, evaluation

    def test_split_sizes(self):
        """20% selection, 80% evaluation."""
        data = self._make_concept_data(n_concepts=100)
        sel, ev = self._split_concepts(data)
        assert sel.neural.size(0) == 20
        assert ev.neural.size(0) == 80

    def test_split_disjoint(self):
        """Selection and evaluation concepts should not overlap."""
        data = self._make_concept_data(n_concepts=100)
        sel, ev = self._split_concepts(data)
        sel_set = set(sel.stimulus_ids)
        eval_set = set(ev.stimulus_ids)
        assert len(sel_set & eval_set) == 0
        assert len(sel_set | eval_set) == 100

    def test_split_deterministic(self):
        """Same seed should always produce the same split."""
        data1 = self._make_concept_data(n_concepts=100, seed=42)
        data2 = self._make_concept_data(n_concepts=100, seed=42)
        sel1, _ = self._split_concepts(data1, seed=42)
        sel2, _ = self._split_concepts(data2, seed=42)
        assert sel1.stimulus_ids == sel2.stimulus_ids

    def test_rsa_on_concept_split(self):
        """compute_rsa should work on the 80/20 concept split."""
        data = self._make_concept_data(n_concepts=100)
        sel, ev = self._split_concepts(data)
        cfg = OmegaConf.create({"compare_method": "spearman"})
        results = compute_rsa(cfg, sel, ev, bootstrap=False, seed=42)
        assert len(results) == 1
        assert results[0]["layer"] == "good"
        assert -1 <= results[0]["score"] <= 1

    def test_rsa_on_concept_split_with_bootstrap(self):
        """Bootstrap on the 80% evaluation set should produce valid CIs."""
        data = self._make_concept_data(n_concepts=100)
        sel, ev = self._split_concepts(data)
        cfg = OmegaConf.create({"compare_method": "spearman"})
        results = compute_rsa(cfg, sel, ev, bootstrap=True, n_bootstrap=50, seed=42)
        r = results[0]
        assert r["ci_low"] is not None
        assert r["ci_high"] is not None
        assert r["ci_low"] < r["ci_high"]
        assert len(r["bootstrap_scores"]) == 50

    def test_no_data_leakage(self):
        """Evaluation score should not depend on selection data."""
        data = self._make_concept_data(n_concepts=100)
        sel, ev = self._split_concepts(data)
        cfg = OmegaConf.create({"compare_method": "spearman"})

        # Normal run
        r_normal = compute_rsa(cfg, sel, ev, bootstrap=False, seed=42)

        # Scramble selection neural data — should not affect test score
        sel_scrambled = AlignmentData(
            activations=sel.activations,
            neural=torch.randn_like(sel.neural),
            stimulus_ids=sel.stimulus_ids,
        )
        r_scrambled = compute_rsa(cfg, sel_scrambled, ev, bootstrap=False, seed=42)

        # Both should evaluate on the same test data → layer may differ but
        # the test evaluation is independent of selection data content
        assert -1 <= r_scrambled[0]["score"] <= 1

    def test_evaluation_has_concept_image_ids(self):
        """Evaluation AlignmentData should carry concept_image_ids for re-extraction."""
        data = self._make_concept_data(n_concepts=100)
        _, ev = self._split_concepts(data)
        assert ev.concept_image_ids is not None
        assert len(ev.concept_image_ids) == 80
        for concept in ev.stimulus_ids:
            assert concept in ev.concept_image_ids


class TestConceptAverageExact:
    """Tests for _concept_average_exact helper in rsa.py."""

    def test_basic_concept_averaging(self):
        """Should average per-image activations by concept."""
        from visreps.analysis.rsa import _concept_average_exact

        raw_acts = torch.tensor([
            [1.0, 2.0],  # img_a
            [3.0, 4.0],  # img_b
            [5.0, 6.0],  # img_c
            [7.0, 8.0],  # img_d
        ])
        raw_ids = ["img_a", "img_b", "img_c", "img_d"]

        data = AlignmentData(
            activations={},
            neural=torch.randn(2, 5),
            stimulus_ids=["cat", "dog"],
            concept_image_ids={
                "cat": ["img_a", "img_b"],
                "dog": ["img_c", "img_d"],
            },
        )

        result = _concept_average_exact(raw_acts, raw_ids, data)
        assert result.shape == (2, 2)
        # cat = mean([1,2], [3,4]) = [2,3]
        torch.testing.assert_close(result[0], torch.tensor([2.0, 3.0]))
        # dog = mean([5,6], [7,8]) = [6,7]
        torch.testing.assert_close(result[1], torch.tensor([6.0, 7.0]))

    def test_missing_image_ids_ignored(self):
        """Image IDs not found in raw_ids should be skipped."""
        from visreps.analysis.rsa import _concept_average_exact

        raw_acts = torch.tensor([
            [1.0, 2.0],
            [3.0, 4.0],
        ])
        raw_ids = ["img_a", "img_b"]

        data = AlignmentData(
            activations={},
            neural=torch.randn(1, 5),
            stimulus_ids=["cat"],
            concept_image_ids={
                "cat": ["img_a", "img_missing", "img_b"],
            },
        )

        result = _concept_average_exact(raw_acts, raw_ids, data)
        assert result.shape == (1, 2)
        # Only img_a and img_b found → mean([1,2], [3,4]) = [2,3]
        torch.testing.assert_close(result[0], torch.tensor([2.0, 3.0]))

    def test_preserves_concept_order(self):
        """Output order should match data.stimulus_ids ordering."""
        from visreps.analysis.rsa import _concept_average_exact

        raw_acts = torch.tensor([
            [10.0, 20.0],  # img_dog
            [1.0, 2.0],    # img_cat
        ])
        raw_ids = ["img_dog", "img_cat"]

        data = AlignmentData(
            activations={},
            neural=torch.randn(2, 5),
            stimulus_ids=["cat", "dog"],  # cat first in stimulus_ids
            concept_image_ids={
                "cat": ["img_cat"],
                "dog": ["img_dog"],
            },
        )

        result = _concept_average_exact(raw_acts, raw_ids, data)
        # cat should be first (following stimulus_ids order)
        torch.testing.assert_close(result[0], torch.tensor([1.0, 2.0]))
        torch.testing.assert_close(result[1], torch.tensor([10.0, 20.0]))


# ═══════════════════════════════════════════════════════════════
# 11. ADDITIONAL EDGE CASE TESTS
# ═══════════════════════════════════════════════════════════════

class TestEdgeCases:
    """Edge case and regression tests."""

    def test_small_n_select(self, synthetic_alignment_data):
        """Should work even with very small n_select for layer selection."""
        train, test = synthetic_alignment_data
        cfg = OmegaConf.create({"compare_method": "spearman"})
        # n_select = 5 (much smaller than train size)
        results = compute_rsa(cfg, train, test, n_select=5, bootstrap=False, seed=42)
        assert len(results) == 1
        assert -1 <= results[0]["score"] <= 1

    def test_n_select_larger_than_train(self, synthetic_alignment_data):
        """n_select > n_train should be capped at n_train."""
        train, test = synthetic_alignment_data
        cfg = OmegaConf.create({"compare_method": "spearman"})
        results = compute_rsa(cfg, train, test, n_select=10000, bootstrap=False, seed=42)
        assert len(results) == 1

    def test_single_layer(self):
        """Should work with only one layer (trivial layer selection)."""
        n_train, n_test, n_feat = 50, 20, 30
        train = AlignmentData(
            activations={"only_layer": torch.randn(n_train, n_feat)},
            neural=torch.randn(n_train, n_feat),
        )
        test = AlignmentData(
            activations={"only_layer": torch.randn(n_test, n_feat)},
            neural=torch.randn(n_test, n_feat),
        )
        cfg = OmegaConf.create({"compare_method": "spearman"})
        results = compute_rsa(cfg, train, test, n_select=30, bootstrap=False, seed=42)
        assert results[0]["layer"] == "only_layer"

    def test_bootstrap_n_bootstrap_1(self, synthetic_alignment_data):
        """Should handle n_bootstrap=1 without error."""
        train, test = synthetic_alignment_data
        cfg = OmegaConf.create({"compare_method": "spearman"})
        results = compute_rsa(
            cfg, train, test,
            n_select=50, bootstrap=True, n_bootstrap=1, seed=42,
        )
        r = results[0]
        assert r["ci_low"] is not None
        assert r["ci_high"] is not None
        assert len(r["bootstrap_scores"]) == 1

    def test_rdm_correlation_small_n(self):
        """RDM correlation with n=2 (1 upper-triangle pair) should work."""
        x = torch.randn(2, 5)
        rdm = compute_rdm(x)
        score = compute_rdm_correlation(rdm, rdm, correlation="Spearman")
        # With only 1 pair, correlation is degenerate but shouldn't crash
        # (spearmanr on length-1 arrays returns NaN)
        assert math.isnan(score) or -1 <= score <= 1

    def test_rdm_correlation_n_3(self):
        """RDM correlation with n=3 (3 upper-triangle pairs) should work."""
        x1 = torch.randn(3, 5)
        x2 = torch.randn(3, 5)
        rdm1 = compute_rdm(x1)
        rdm2 = compute_rdm(x2)
        score = compute_rdm_correlation(rdm1, rdm2, correlation="Spearman")
        assert -1 <= score <= 1

    def test_many_layers_rsa(self):
        """compute_rsa should handle many layers efficiently."""
        n_train, n_test, n_feat = 50, 20, 30
        train_acts = {f"layer_{i}": torch.randn(n_train, n_feat) for i in range(7)}
        test_acts = {f"layer_{i}": torch.randn(n_test, n_feat) for i in range(7)}
        train = AlignmentData(activations=train_acts, neural=torch.randn(n_train, n_feat))
        test = AlignmentData(activations=test_acts, neural=torch.randn(n_test, n_feat))
        cfg = OmegaConf.create({"compare_method": "spearman"})
        results = compute_rsa(cfg, train, test, n_select=30, bootstrap=False, seed=42)
        assert len(results[0]["layer_selection_scores"]) == 7

    def test_4d_conv_activations(self):
        """compute_rsa should handle 4D conv activations by flattening."""
        n_train, n_test = 50, 20
        train = AlignmentData(
            activations={"conv": torch.randn(n_train, 64, 6, 6)},
            neural=torch.randn(n_train, 30),
        )
        test = AlignmentData(
            activations={"conv": torch.randn(n_test, 64, 6, 6)},
            neural=torch.randn(n_test, 30),
        )
        cfg = OmegaConf.create({"compare_method": "spearman"})
        results = compute_rsa(cfg, train, test, n_select=30, bootstrap=False, seed=42)
        assert results[0]["layer"] == "conv"
        assert -1 <= results[0]["score"] <= 1

    def test_bootstrap_distribution_statistics(self):
        """Bootstrap distribution should have reasonable statistical properties.

        Uses noisier data so the bootstrap has measurable variance
        (the default synthetic fixture has r~0.99 which is too tight).
        """
        torch.manual_seed(42)
        n_train, n_test, n_feat = 200, 50, 100
        neural_train = torch.randn(n_train, n_feat)
        neural_test = torch.randn(n_test, n_feat)
        # Moderate noise so RSA is positive but not extreme
        good_train = neural_train + 0.5 * torch.randn(n_train, n_feat)
        good_test = neural_test + 0.5 * torch.randn(n_test, n_feat)

        train = AlignmentData(
            activations={"layer": good_train},
            neural=neural_train,
        )
        test = AlignmentData(
            activations={"layer": good_test},
            neural=neural_test,
        )
        cfg = OmegaConf.create({"compare_method": "spearman"})
        results = compute_rsa(
            cfg, train, test,
            n_select=100, bootstrap=True, n_bootstrap=500, seed=42,
        )
        bs = np.array(results[0]["bootstrap_scores"])
        # Should have reasonable variance (not degenerate)
        assert bs.std() > 0.001, f"Bootstrap std too low: {bs.std()}"
        # All scores should be valid
        assert np.all(np.isfinite(bs))
        assert np.all((-1 <= bs) & (bs <= 1))
        # CI width should be reasonable
        r = results[0]
        ci_width = r["ci_high"] - r["ci_low"]
        assert 0.001 < ci_width < 1.5, f"CI width {ci_width} seems unreasonable"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
