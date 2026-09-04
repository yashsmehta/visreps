"""Feature extraction, layer naming, and config validation (no weights downloaded)."""
import pytest
import torch
import torchvision.models as tv_models
from omegaconf import OmegaConf

from visreps.evals import _listify, _set_torchvision_cfg
from visreps.models import standard_model
from visreps.models.custom_model import CustomCNN
from visreps.models.utils import FeatureExtractor, TORCHVISION_RETURN_NODES
from visreps.utils import ConfigVerifier, get_seed_letter


def _extract(model, nodes):
    fe = FeatureExtractor(model.eval(), {n: n for n in nodes}, extract_pre_and_post=True)
    with torch.no_grad():
        return fe(torch.randn(2, 3, 224, 224))


# ───────────────────────── feature extraction ─────────────────────────
@pytest.mark.parametrize("model", [CustomCNN(num_classes=8), tv_models.alexnet()])
def test_pre_activations_are_not_overwritten_by_inplace_relu(model):
    out = _extract(model, ["conv1", "fc1"])
    for layer in ["conv1", "fc1"]:
        assert out[f"{layer}_pre"].min() < 0, f"{layer}_pre looks post-ReLU"
        assert out[f"{layer}_post"].min() >= 0
        assert not torch.equal(out[f"{layer}_pre"], out[f"{layer}_post"])


@pytest.mark.parametrize("model_name", ["AlexNet", "VGG16", "ResNet50", "ViTBase", "ConvNeXt_Base"])
def test_every_declared_return_node_produces_output(model_name):
    model = getattr(standard_model, model_name)(pretrained_dataset="none")
    nodes = TORCHVISION_RETURN_NODES[model_name]
    out = _extract(model, nodes)
    for node in nodes:
        assert any(k in out for k in (node, f"{node}_pre", f"{node}_post")), f"{node} missing: {list(out)}"
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
    fe = FeatureExtractor(model.eval(), {n: n for n in ["conv1", "conv5", "fc2"]})
    assert set(fe.layer_mapping) == {"conv1_pre", "conv1_post", "conv5_pre", "conv5_post", "fc2_pre", "fc2_post"}
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
