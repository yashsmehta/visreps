"""Feature extraction, layer naming, and config validation (no weights downloaded)."""
import pytest
import torch
from omegaconf import OmegaConf

from visreps.evals import _listify, _set_torchvision_cfg
from visreps.models import standard_model
from visreps.models.custom_model import CustomCNN
from visreps.models.utils import FeatureExtractor, TORCHVISION_RETURN_NODES
from visreps.utils import ConfigVerifier, get_seed_letter


def _extract(model, nodes):
    fe = FeatureExtractor(model.eval(), {n: n for n in nodes})
    with torch.no_grad():
        return fe(torch.randn(2, 3, 224, 224))


# ───────────────────────── feature extraction ─────────────────────────
@pytest.mark.parametrize("normalization", [True, False])
def test_features_are_post_normalization_and_activation(normalization):
    from torch import nn
    model = nn.Module()
    norm = nn.BatchNorm2d(2) if normalization else nn.Identity()
    if normalization:
        with torch.no_grad():
            norm.running_mean.fill_(3)
            norm.running_var.fill_(4)
            norm.weight.fill_(2)
            norm.bias.fill_(-1)
    model.features = nn.Sequential(nn.Conv2d(3, 2, 1), norm, nn.ReLU(inplace=True))
    model.classifier = nn.Sequential(nn.Flatten(), nn.Linear(8, 2))
    model.forward = lambda x: model.classifier(model.features(x))
    model.eval()
    x = torch.randn(2, 3, 2, 2)
    expected = norm(model.features[0](x)).detach().clone()
    extractor = FeatureExtractor(model, {"conv1": "conv1"})
    out = extractor(x)
    assert set(out) == {"conv1"}
    torch.testing.assert_close(out["conv1"], expected.relu())


@pytest.mark.parametrize("model_name", ["AlexNet", "VGG16", "ResNet50", "ViTBase", "ConvNeXt_Base"])
def test_every_declared_return_node_produces_output(model_name):
    model = getattr(standard_model, model_name)(pretrained_dataset="none")
    nodes = TORCHVISION_RETURN_NODES[model_name]
    out = _extract(model, nodes)
    for node in nodes:
        assert node in out, f"{node} missing: {list(out)}"
    assert all(t.shape[0] == 2 for t in out.values())


@pytest.mark.parametrize("model_name,timm_id", [
    ("DINOv2_ViT_B14", "vit_base_patch14_dinov2"), ("DINOv3_ViT_L16", "vit_large_patch16_dinov3"),
])
def test_timm_extractor_return_nodes(model_name, timm_id):
    import timm
    from visreps.models.standard_model import TimmViTExtractor
    model = TimmViTExtractor.__new__(TimmViTExtractor)
    torch.nn.Module.__init__(model)
    model.model = timm.create_model(timm_id, pretrained=False, num_classes=0, dynamic_img_size=True).float()
    model.return_nodes = {n: n for n in TORCHVISION_RETURN_NODES[model_name]}
    with torch.no_grad():
        out = model.eval()(torch.randn(2, 3, 224, 224))
    assert set(TORCHVISION_RETURN_NODES[model_name]) <= set(out)


def test_custom_cnn_layer_names_and_forward():
    model = CustomCNN(num_classes=10)
    nodes = TORCHVISION_RETURN_NODES["CustomCNN"]
    fe = FeatureExtractor(model.eval(), {n: n for n in nodes})
    with torch.no_grad():
        out = fe(torch.randn(2, 3, 224, 224))
    assert len(out) == 7
    assert set(out) == set(nodes)
    assert all((value >= 0).all() for value in out.values())
    with torch.no_grad():
        assert model(torch.randn(2, 3, 224, 224)).shape == (2, 10)


# ───────────────────────── config validation ─────────────────────────
def _eval_cfg(**overrides):
    base = {
        "mode": "eval", "seed": 1, "neural_dataset": "nsd", "analysis": "rsa",
        "compare_method": "spearman", "load_model_from": "checkpoint",
        "subject_idx": [0], "region": ["ventral visual stream"],
        "checkpoint_dir": "/data/ymehta3/default", "cfg_id": 1000,
        "checkpoint_model": "checkpoint_epoch_20.pth", "model_name": "CustomCNN", "verbose": False,
    }
    base.update(overrides)
    return OmegaConf.create(base)


def test_verifier_normalizes_scalars_to_lists():
    cfg = ConfigVerifier(_eval_cfg(subject_idx=3, region="early visual stream")).verify()
    assert list(cfg.subject_idx) == [3] and list(cfg.region) == ["early visual stream"]


def test_verifier_things_forces_na_and_encoding_forces_pearson():
    cfg = ConfigVerifier(_eval_cfg(neural_dataset="things-behavior")).verify()
    assert cfg.region == "N/A" and cfg.subject_idx == "N/A"
    cfg = ConfigVerifier(_eval_cfg(analysis="encoding_score", compare_method="kendall")).verify()
    assert cfg.compare_method == "pearson"


@pytest.mark.parametrize("overrides,match", [
    ({"seed": 5}, "Invalid seed"),
    ({"region": ["nonexistent"]}, "Invalid region"),
    ({"subject_idx": [8]}, "Invalid subject"),
    ({"neural_dataset": "tvsd", "region": ["V2"]}, "Invalid region"),
    ({"analysis": "cka"}, "Invalid analysis"),
    ({"compare_method": "cosine"}, "Invalid compare_method"),
    ({"neural_dataset": "things-behavior", "analysis": "encoding_score"}, "encoding_score"),
])
def test_verifier_rejects_invalid_config(overrides, match):
    with pytest.raises(AssertionError, match=match):
        ConfigVerifier(_eval_cfg(**overrides)).verify()


def test_eval_helpers():
    assert [get_seed_letter(s) for s in (1, 2, 3)] == ["a", "b", "c"]
    assert _listify(0) == [0] and _listify(["a"]) == ["a"]
    cfg = _set_torchvision_cfg(OmegaConf.create({"pretrained_dataset": "imagenet1k"}))
    assert (cfg.cfg_id, cfg.epoch) == ("pretrained", -1)
    assert _set_torchvision_cfg(OmegaConf.create({"pretrained_dataset": "none"})).cfg_id == "untrained"
