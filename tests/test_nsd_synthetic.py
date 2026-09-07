"""NSD-synthetic: reference input transform, reliable-voxel archive, layer reuse."""
import pickle
import sqlite3

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from visreps.evals import _get_eval_transform, _lookup_nsd_best_layers, _split_synthetic_stimuli
from visreps.dataloaders.neural import (NsdSyntheticTransform, load_all_nsd_synthetic_data,
                                        _BLANK_AFTER_CROP)

DATA = "datasets/neural/nsd_synthetic/nsd_synthetic_data.pkl"
STIMULI = "datasets/neural/nsd_synthetic/stimuli"
REGIONS = ["early visual stream", "ventral visual stream"]


@pytest.fixture(scope="module")
def synthetic():
    try:
        with open(DATA, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        pytest.skip(f"{DATA} not built on this machine")


def test_transform_squares_before_resizing():
    """The 714x1360 frame is cropped to its content square, not squashed.

    Resize-then-crop clips a band off every side and truncates the word-position
    stimuli, which is what this transform exists to avoid.
    """
    raw = Image.open(f"{STIMULI}/word4_pos2_1.png")
    assert raw.size == (1360, 714)

    out = NsdSyntheticTransform()(raw)
    assert out.shape == (3, 224, 224) and torch.isfinite(out).all()

    # sqrt linearisation, centre crop to the 714x714 square, then resize.
    linear = (np.sqrt(np.asarray(raw.convert("RGB")) / 255) * 255).astype(np.uint8)
    square = Image.fromarray(linear).crop((323, 0, 1037, 714))
    expected = torch.from_numpy(
        np.asarray(square.resize((224, 224), Image.Resampling.BILINEAR)).copy()
    ).permute(2, 0, 1).float() / 255
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
    torch.testing.assert_close(out, (expected - mean) / std, rtol=0, atol=0)


def test_only_nsd_synthetic_gets_the_reference_transform():
    for dataset in ["nsd", "tvsd", "things-behavior", "nsd_synthetic"]:
        cfg = OmegaConf.create({"neural_dataset": dataset, "model_name": "CustomCNN"})
        is_reference = isinstance(_get_eval_transform(cfg), NsdSyntheticTransform)
        assert is_reference == (dataset == "nsd_synthetic"), dataset


def test_archive_holds_only_reliable_voxels(synthetic):
    assert synthetic["ncsnr_threshold"] == 0.2
    for region, subjects in synthetic["data"].items():
        for subj, responses in subjects.items():
            assert (responses.ncsnr.values > 0.2).all(), (region, subj)
            assert np.isfinite(responses.values).all(), (region, subj)
            assert responses.sizes["stimulus"] == 220


def test_loader_matches_nsd_subjects_and_region_names(synthetic):
    cfg = OmegaConf.create({"neural_dataset": "nsd_synthetic"})
    data = load_all_nsd_synthetic_data(cfg, subjects=[0, 1], regions=REGIONS)

    assert data["regions"] == REGIONS and data["subjects"] == [0, 1]
    assert len(data["test_ids"]) == 204 and not set(data["test_ids"]) & _BLANK_AFTER_CROP
    assert set(data["stimuli"]) == set(data["test_ids"])
    for region, key in zip(REGIONS, ["early", "ventral"]):
        for subj in [0, 1]:
            responses = data["neural"][region][subj]
            assert set(responses) == set(data["test_ids"])
            n_voxels = synthetic["data"][key][subj].sizes["neuroid"]
            assert next(iter(responses.values())).shape == (n_voxels,)


@pytest.fixture
def nsd_results(tmp_path, monkeypatch):
    """A results.db holding one regular-NSD RSA run: conv4 for every subject."""
    import visreps.utils as vu

    db = tmp_path / "results.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE results (run_id TEXT, compare_method TEXT, layer TEXT, "
                     "analysis TEXT, seed INTEGER, epoch INTEGER, region TEXT, "
                     "subject_idx TEXT, neural_dataset TEXT, cfg_id INTEGER, "
                     "model_name TEXT, checkpoint_dir TEXT, reconstruct_from_pcs BOOLEAN)")
        conn.executemany(
            "INSERT INTO results VALUES ('r','spearman',?,'rsa',1,20,?,?,'nsd',32,"
            "'CustomCNN','/ckpt',0)",
            [("conv4", REGIONS[0], str(s)) for s in range(8)])
    monkeypatch.setattr(vu, "_RESULTS_DB_PATH", db)
    return OmegaConf.create(dict(
        neural_dataset="nsd_synthetic", analysis="rsa", compare_method="spearman",
        checkpoint_dir="/ckpt", cfg_id=32, seed=1, epoch=20,
        model_name="CustomCNN", reconstruct_from_pcs=False))


def test_layer_lookup_reuses_the_matching_nsd_run(nsd_results):
    assert _lookup_nsd_best_layers(nsd_results, list(range(8)), [REGIONS[0]]) == {REGIONS[0]: "conv4"}


def test_layer_lookup_fails_loudly_without_an_nsd_run(nsd_results):
    """A synthetic run must not silently invent a layer when NSD has not been run."""
    missing = OmegaConf.merge(nsd_results, {"seed": 99})
    with pytest.raises(ValueError, match="No regular-NSD RSA result"):
        _lookup_nsd_best_layers(missing, list(range(8)), [REGIONS[0]])


def test_layer_lookup_rejects_disagreeing_subjects(nsd_results, tmp_path):
    """One layer per ROI: subject-level disagreement means the NSD run is stale."""
    with sqlite3.connect(tmp_path / "results.db") as conn:
        conn.execute("UPDATE results SET layer='fc1' WHERE subject_idx='3'")
    with pytest.raises(ValueError, match="disagree on the layer"):
        _lookup_nsd_best_layers(nsd_results, list(range(8)), [REGIONS[0]])


def test_blank_stimuli_are_the_words_outside_the_content_square():
    """The excluded images really are spatially constant after the reference crop."""
    assert len(_BLANK_AFTER_CROP) == 16
    for sid in sorted(_BLANK_AFTER_CROP):
        out = NsdSyntheticTransform()(Image.open(f"{STIMULI}/{sid}.png"))
        assert (out - out[:, :1, :1]).abs().max() == 0, sid


def test_split_is_stratified_and_fixed(synthetic):
    ids = [s for s in synthetic["shared_stimulus_names"] if s not in _BLANK_AFTER_CROP]
    select, test = _split_synthetic_stimuli(ids)
    assert len(select) == len(test) == 102 and not set(select) & set(test)
    assert sorted(select + test) == sorted(ids)
    for family in {s.rsplit("_", 1)[0] for s in ids}:
        n_sel = sum(s.startswith(family + "_") for s in select)
        n_test = sum(s.startswith(family + "_") for s in test)
        assert n_sel == n_test, family
    assert _split_synthetic_stimuli(ids) == (select, test)  # deterministic
