"""Check TVSD reliability measures against independent implementations and the dataset."""

import numpy as np
import pytest
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr

from scripts.preprocess_data.preprocess_tvsd import mean_reliability, oracle_correlation


def test_oracle_matches_leave_one_out_pearson():
    rng = np.random.default_rng(42)
    responses = rng.normal(size=(100, 30, 4)) + rng.normal(size=(100, 1, 4))
    expected = [
        np.mean([
            pearsonr(responses[:, repeat, electrode],
                     np.delete(responses[:, :, electrode], repeat, axis=1).mean(axis=1)).statistic
            for repeat in range(30)
        ])
        for electrode in range(4)
    ]
    np.testing.assert_allclose(oracle_correlation(responses), expected, atol=1e-12)


def test_oracle_perfect_and_constant_electrodes():
    responses = np.repeat(np.arange(100.0)[:, None, None], 30, axis=1)
    np.testing.assert_allclose(oracle_correlation(responses), 1)
    assert np.isnan(oracle_correlation(np.ones((100, 30, 1)))).all()


def test_mean_reliability_matches_pairwise_correlations():
    """Same quantity as the authors' `1 - pdist(reps, 'correlation')`, averaged."""
    rng = np.random.default_rng(0)
    responses = rng.normal(size=(100, 30, 3)) + rng.normal(size=(100, 1, 3))
    expected = [
        (1 - pdist(responses[:, :, electrode].T, "correlation")).mean()
        for electrode in range(3)
    ]
    np.testing.assert_allclose(mean_reliability(responses), expected, atol=1e-12)


@pytest.mark.parametrize("monkey", ["F", "N"])
def test_reliability_and_oracle_match_dataset_fields(monkey):
    """Our measures reproduce the `reliab` and `oracle` fields shipped with TVSD."""
    mat73 = pytest.importorskip("mat73")
    data = pytest.importorskip("bonner.datasets.papale2025_tvsd._data")
    raw = mat73.loadmat(data.download(monkey=monkey, normalized=True))
    reps = np.asarray(raw["test_MUA_reps"]).transpose(1, 2, 0)  # stimulus × repetition × electrode
    np.testing.assert_allclose(mean_reliability(reps), np.nanmean(raw["reliab"], axis=1), atol=1e-10)
    np.testing.assert_allclose(oracle_correlation(reps), np.asarray(raw["oracle"]), atol=1e-10)
