"""Shared pytest setup: no test may write to results.db or the SRP cache."""
import os
import sys

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture(autouse=True)
def _isolate_side_effects(tmp_path, monkeypatch):
    """Redirect results.db and the SRP transformer cache to a temp dir for every test."""
    import visreps.utils as vu

    monkeypatch.setattr(vu, "_RESULTS_DB_PATH", tmp_path / "results.db")
    monkeypatch.setenv("VISREPS_SRP_CACHE", str(tmp_path / "srp_cache"))
