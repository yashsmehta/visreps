"""Guard the dataset and architecture controls in the matched AlexNet runs."""
from pathlib import Path

import pytest
import torch
from torchvision import models, transforms

from visreps.dataloaders import obj_cls, obj_cls_folder
from visreps.models.utils import load_model
from visreps.utils import load_config, validate_config


def config(version):
    root = Path(__file__).resolve().parents[1] / "experiments/alexnet_dataset_comparison"
    return validate_config(load_config([root / "common.json", root / f"imagenet{version}.json"]))


def test_matched_recipe_and_exact_torchvision_architecture():
    cfg = config(2010)
    other = config(2012)
    differences = {k for k in cfg if cfg[k] != other[k]}
    assert differences == {"imagenet_version", "dataset_path", "checkpoint_dir"}
    torch.manual_seed(1)
    actual = load_model(cfg, torch.device("cpu"), num_classes=1000)
    torch.manual_seed(1)
    expected = models.alexnet(weights=None, dropout=0.3)
    assert repr(actual) == repr(expected)
    assert not any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm) for m in actual.modules())
    for key, value in actual.state_dict().items():
        torch.testing.assert_close(value, expected.state_dict()[key], rtol=0, atol=0)


def test_explicit_folder_routing_even_with_parquet_installed(monkeypatch):
    monkeypatch.setattr(obj_cls, "ImageNetParquet", object())
    monkeypatch.setattr(obj_cls_folder, "prepare_imgnet_data", lambda *args: "folder")
    assert obj_cls.prepare_imgnet_data(config(2010), False, True, True, True) == "folder"


def test_missing_2012_path_fails_without_loading_2010():
    cfg = config(2012)
    cfg.dataset_path = None
    with pytest.raises(ValueError, match="Set dataset_path"):
        obj_cls.prepare_imgnet_data(cfg, False, True, True, True)


@pytest.mark.parametrize("mismatch", [False, True])
def test_2012_official_splits_and_independent_label_map(monkeypatch, tmp_path, mismatch):
    class FakeImageFolder:
        def __init__(self, root, transform):
            self.root, self.transform = root, transform
            self.classes = [f"n{i:08d}" for i in range(1000)]
            self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
            if mismatch and root.name == "val":
                self.class_to_idx[self.classes[0]] = 1

        def __len__(self):
            return 1000

    monkeypatch.setattr("torchvision.datasets.ImageFolder", FakeImageFolder)
    cfg = config(2012)
    cfg.dataset_path, cfg.num_workers = str(tmp_path), 0
    if mismatch:
        with pytest.raises(ValueError, match="mappings differ"):
            obj_cls.prepare_imgnet_data(cfg, False, True, True, True)
        return
    datasets, _ = obj_cls.prepare_imgnet_data(cfg, False, True, True, True)
    assert datasets["train"].root == tmp_path / "train"
    assert datasets["test"].root == tmp_path / "val"
    assert any(isinstance(t, transforms.RandomRotation) for t in datasets["train"].transform.transforms)
    assert not any(isinstance(t, transforms.RandomRotation) for t in datasets["test"].transform.transforms)
