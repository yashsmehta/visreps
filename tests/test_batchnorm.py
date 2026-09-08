"""Calibration must not adapt to held-out images or change learned parameters."""
import copy

import numpy as np
import pytest
import torch
from torch import nn
from omegaconf import OmegaConf
from torchvision.transforms import ToTensor

from visreps.dataloaders.neural import _make_loader
from visreps.models.batchnorm import prepare_eval_batchnorm, training_image_ids
from visreps.utils import _compute_run_id


def test_excludes_test_images_across_subjects():
    neural = {"V1": {0: {"train": {"a": 0, "b": 0}, "test": {"c": 0}},
                     1: {"train": {"a": 0, "c": 0}, "test": {"b": 0}}}}
    assert training_image_ids(neural, ["a", "b", "c"]) == ["a"]


def test_calibration_and_cache(tmp_path):
    torch.manual_seed(1)
    model = nn.Sequential(nn.Conv2d(3, 2, 1), nn.BatchNorm2d(2), nn.ReLU(),
                          nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.9),
                          nn.Linear(2, 3), nn.BatchNorm1d(3))
    original = copy.deepcopy(model)
    images = {str(i): np.full((4, 4, 3), i * 20, dtype=np.uint8) for i in range(6)}
    loader = _make_loader(images, ToTensor(), 2, 0)
    cfg = OmegaConf.create(dict(neural_dataset="nsd", bn_cache_dir=str(tmp_path),
                                bn_calibration_batchsize=2))
    old_id = _compute_run_id(cfg)
    dropout_modes = []
    hook = model[5].register_forward_pre_hook(lambda m, _: dropout_modes.append(m.training))
    prepare_eval_batchnorm(model, cfg, loader, list(images)[:5], "cpu")
    hook.remove()
    assert dropout_modes == [False, False]  # singleton merged: batches of 2 and 3
    assert not any(m.training for m in model.modules())
    for before, after in zip(original.parameters(), model.parameters()):
        assert torch.equal(before, after)
    assert not torch.equal(model[1].running_mean, original[1].running_mean)
    assert model[1].momentum == original[1].momentum
    assert _compute_run_id(cfg) != old_id

    reloaded = copy.deepcopy(original)
    # A cache hit must perform no forward pass.
    def fail(*args):
        raise AssertionError("Unexpected calibration forward")
    hook = reloaded.register_forward_pre_hook(fail)
    prepare_eval_batchnorm(reloaded, cfg, loader, list(images)[:5], "cpu")
    hook.remove()
    x = torch.randn(3, 3, 4, 4)
    torch.testing.assert_close(model(x), reloaded(x), rtol=0, atol=0)

    # Changing an excluded test image cannot affect calibration.
    images["5"][:] = 255
    cfg.bn_cache_dir = str(tmp_path / "fresh")
    fresh = copy.deepcopy(original)
    prepare_eval_batchnorm(fresh, cfg, loader, list(images)[:5], "cpu")
    torch.testing.assert_close(model(x), fresh(x), rtol=0, atol=0)

    identity = cfg.bn_calibration
    prepare_eval_batchnorm(copy.deepcopy(original), cfg, loader, list(images)[:4], "cpu")
    assert cfg.bn_calibration != identity


def _bn_model():
    torch.manual_seed(1)
    return nn.Sequential(nn.Conv2d(3, 2, 1), nn.BatchNorm2d(2), nn.ReLU(),
                         nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(2, 3))


def test_checkpoint_source_keeps_running_stats(tmp_path):
    model = _bn_model()
    model[1].running_mean.fill_(0.5)
    cfg = OmegaConf.create(dict(neural_dataset="nsd", bn_cache_dir=str(tmp_path),
                                bn_calibration_source="checkpoint"))
    prepare_eval_batchnorm(model, cfg, None, [], "cpu")
    assert torch.all(model[1].running_mean == 0.5)
    assert cfg.bn_calibration == "checkpoint"
    assert not any(m.training for m in model.modules())


def test_invalid_source(tmp_path):
    cfg = OmegaConf.create(dict(neural_dataset="nsd", bn_calibration_source="things"))
    with pytest.raises(ValueError, match="bn_calibration_source"):
        prepare_eval_batchnorm(_bn_model(), cfg, None, [], "cpu")


def test_imagenet_source_ignores_dataset_images(tmp_path, monkeypatch):
    """ImageNet calibration uses a fixed ImageNet subset, never the neural stimuli."""
    import visreps.dataloaders.obj_cls as obj_cls
    from torch.utils.data import TensorDataset

    torch.manual_seed(0)
    imagenet = TensorDataset(torch.randn(20, 3, 4, 4) * 5 + 3, torch.zeros(20, dtype=torch.long))
    calls = []

    def fake_loader(cfg, shuffle=True, **kw):
        calls.append((dict(cfg), shuffle))
        return {"train": imagenet, "test": imagenet}, {}
    monkeypatch.setattr(obj_cls, "get_obj_cls_loader", fake_loader)

    model = _bn_model()
    original = copy.deepcopy(model)
    cfg = OmegaConf.create(dict(neural_dataset="things-behavior", bn_cache_dir=str(tmp_path),
                                bn_calibration_source="imagenet", bn_calibration_images=10,
                                bn_calibration_batchsize=5, num_workers=0))
    seen = []
    model.register_forward_pre_hook(lambda m, inp: seen.append(inp[0].shape[0]))
    prepare_eval_batchnorm(model, cfg, None, [], "cpu")
    assert calls and calls[0][1] is False and calls[0][0]["pca_labels"] is False
    assert seen == [5, 5]  # 10 images in batches of 5
    for before, after in zip(original.parameters(), model.parameters()):
        assert torch.equal(before, after)
    assert not torch.equal(model[1].running_mean, original[1].running_mean)
    assert cfg.bn_calibration not in ("none", "checkpoint")
    assert (tmp_path / f"imagenet_{cfg.bn_calibration}.pt").exists()

    # Cache hit: identical buffers, no forward pass, distinct from a dataset-source run.
    reloaded = copy.deepcopy(original)
    reloaded.register_forward_pre_hook(lambda m, inp: pytest.fail("Unexpected forward"))
    cfg2 = OmegaConf.create(dict(cfg))
    prepare_eval_batchnorm(reloaded, cfg2, None, [], "cpu")
    assert cfg2.bn_calibration == cfg.bn_calibration
    torch.testing.assert_close(reloaded[1].running_var, model[1].running_var, rtol=0, atol=0)


def test_no_batchnorm_and_empty_training_set(tmp_path):
    cfg = OmegaConf.create(dict(neural_dataset="nsd", bn_cache_dir=str(tmp_path)))
    model = nn.Linear(2, 2)
    assert prepare_eval_batchnorm(model, cfg, None, [], "cpu") is model
    assert cfg.bn_calibration == "none"
    with pytest.raises(ValueError, match="at least two"):
        prepare_eval_batchnorm(nn.BatchNorm1d(2), cfg, None, [], "cpu")


def test_things_calibrates_only_layer_selection_images(tmp_path, monkeypatch):
    import visreps.evals as ev

    images = {str(i): np.full((4, 4, 3), i, dtype=np.uint8) for i in range(10)}
    loader = _make_loader(images, ToTensor(), 2, 0)
    neural = {"image_ids": {f"concept{i}": [str(i)] for i in range(10)},
              "embeddings": {f"concept{i}": np.ones(3) for i in range(10)}}
    cfg = OmegaConf.create(dict(load_model_from="torchvision", model_name="CustomCNN",
                                neural_dataset="things-behavior", seed=1, log_expdata=False))
    model = nn.Linear(2, 2)
    monkeypatch.setattr(ev, "get_neural_loader", lambda cfg: (neural, loader))
    monkeypatch.setattr(ev.mutils, "load_model", lambda *a, **kw: model)
    monkeypatch.setattr(ev.mutils, "configure_feature_extractor", lambda cfg, m, **kw: m)
    calibrated = set()

    def calibrate(model, cfg, dl, image_ids, device):
        calibrated.update(image_ids)
    monkeypatch.setattr(ev, "prepare_eval_batchnorm", calibrate)
    monkeypatch.setattr(ev.mutils, "get_activations",
                        lambda *a: ({"fc2": torch.randn(10, 3)}, list(images)))

    def score(cfg, selection, evaluation, **kwargs):
        selection_ids = {sid for c in selection.stimulus_ids for sid in neural["image_ids"][c]}
        test_ids = {sid for c in evaluation.stimulus_ids for sid in neural["image_ids"][c]}
        assert calibrated == selection_ids
        assert len(calibrated) == 2
        assert calibrated.isdisjoint(test_ids)
        return []
    monkeypatch.setattr(ev, "compute_traintest_alignment", score)
    ev.eval(cfg)
