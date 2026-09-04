"""End-to-end evals on real data (GPU + checkpoint required). Run with: pytest -m slow

Writes go to a temp results.db (see conftest.py). Each test checks the score is
sane and that one layer was selected per ROI across subjects.
"""
import os
import sqlite3

import pandas as pd
import pytest
import torch

CHECKPOINT = "/data/ymehta3/default/cfg1000a/checkpoint_epoch_20.pth"
pytestmark = pytest.mark.slow


@pytest.fixture(autouse=True)
def _require_gpu_and_checkpoint():
    if not torch.cuda.is_available():
        pytest.skip("No GPU")
    if not os.path.exists(CHECKPOINT):
        pytest.skip(f"Missing {CHECKPOINT}")


def _run(*overrides):
    from dotenv import load_dotenv
    load_dotenv()
    import visreps.utils as utils
    import visreps.evals as evals

    base = [
        "mode=eval", "cfg_id=1000", "seed=1", "checkpoint_dir=/data/ymehta3/default",
        "checkpoint_model=checkpoint_epoch_20.pth", "log_expdata=true",
        "bootstrap=true", "n_bootstrap=20", "verbose=false", "batchsize=64", "num_workers=4",
    ]
    cfg = utils.validate_config(utils.load_config("configs/eval/base.json", base + list(overrides)))
    return evals.eval(cfg)


def _assert_sane(results, n_rows):
    assert len(results) == n_rows
    assert (results["score"] > 0).all() and (results["ci_low"] <= results["score"]).all()
    assert (results["score"] <= results["ci_high"]).all()
    import visreps.utils as vu
    assert len(pd.read_sql("SELECT * FROM results", sqlite3.connect(vu._RESULTS_DB_PATH))) == n_rows


def test_nsd_rsa_one_layer_per_roi():
    res = _run("neural_dataset=nsd", "analysis=rsa", "compare_method=spearman",
               "subject_idx=[0,1]", 'region=["ventral visual stream"]')
    _assert_sane(res, 2)
    assert res["layer"].nunique() == 1


def test_tvsd_encoding_one_layer_per_roi():
    res = _run("neural_dataset=tvsd", "analysis=encoding_score", "subject_idx=[0,1]", "region=[IT]")
    _assert_sane(res, 2)
    assert res["layer"].nunique() == 1 and (res["compare_method"] == "pearson").all()


def test_things_rsa():
    res = _run("neural_dataset=things-behavior", "analysis=rsa", "compare_method=spearman")
    _assert_sane(res, 1)
    assert res.iloc[0]["score"] > 0.1
