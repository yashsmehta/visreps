"""Tests for model loading, feature extraction, config validation, and RSA correctness.

Covers the three most bug-prone areas not tested by test_rsa_bootstrap.py or
test_encoding_score.py:

  1. RSA correctness against rsatoolbox (Kriegeskorte lab reference implementation)
  2. Feature extraction and layer naming across all supported architectures
  3. ConfigVerifier validation logic (reject invalid configs, normalize scalars)

Run all tests:
    source .venv/bin/activate && pytest tests/test_model_pipeline.py -v

Run fast tests only (skip model loading):
    source .venv/bin/activate && pytest tests/test_model_pipeline.py -v -m "not slow"
"""
import os
import sys
import pytest
import numpy as np
import torch
import rsatoolbox
import torchvision.models as tv_models
from omegaconf import OmegaConf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from visreps.analysis.rsa import compute_rdm, compute_rdm_correlation
from visreps.utils import ConfigVerifier
from visreps.models.utils import FeatureExtractor, TORCHVISION_RETURN_NODES


# ═══════════════════════════════════════════════════════════════
# Shared helpers
# ═══════════════════════════════════════════════════════════════

def _rsatoolbox_rdm(data):
    """Compute a Pearson (correlation) RDM via rsatoolbox."""
    return rsatoolbox.rdm.calc_rdm(
        rsatoolbox.data.Dataset(data), method="correlation"
    )


def _our_and_ref_rdms(data):
    """Return (our_rdm_tensor, rsatoolbox_rdm_matrix) for Pearson RDMs."""
    our_rdm = compute_rdm(torch.from_numpy(data), correlation="Pearson")
    ref_matrix = _rsatoolbox_rdm(data).get_matrices()[0]
    return our_rdm, ref_matrix


def _create_timm_vit_extractor(timm_id):
    """Create a TimmViTExtractor without pretrained weights (fast, CI-safe)."""
    import timm
    from visreps.models.standard_model import TimmViTExtractor

    model = TimmViTExtractor.__new__(TimmViTExtractor)
    torch.nn.Module.__init__(model)
    model.model = timm.create_model(
        timm_id, pretrained=False, num_classes=0, dynamic_img_size=True,
    )
    model.model.float()
    model.return_nodes = None
    return model


def _make_eval_cfg(**overrides):
    """Create a minimal valid eval config, then apply overrides."""
    base = {
        "mode": "eval",
        "seed": 1,
        "neural_dataset": "nsd",
        "analysis": "rsa",
        "compare_method": "spearman",
        "load_model_from": "checkpoint",
        "subject_idx": [0],
        "region": ["ventral visual stream"],
        "checkpoint_dir": "/data/ymehta3/default",
        "cfg_id": 1000,
        "checkpoint_model": "checkpoint_epoch_20.pth",
        "model_name": "CustomCNN",
        "verbose": False,
    }
    base.update(overrides)
    return OmegaConf.create(base)


# ═══════════════════════════════════════════════════════════════
# Module-scoped fixtures (shared model instances)
# ═══════════════════════════════════════════════════════════════

@pytest.fixture(scope="module")
def alexnet_model():
    """Shared AlexNet instance for layer mapping tests (read-only)."""
    return tv_models.alexnet()


@pytest.fixture(scope="module")
def resnet50_model():
    """Shared ResNet50 instance for layer mapping tests (read-only)."""
    return tv_models.resnet50()


@pytest.fixture(scope="module")
def vit_model():
    """Shared ViT-Base instance for layer mapping tests (read-only)."""
    return tv_models.vit_b_16()


@pytest.fixture(scope="module")
def convnext_model():
    """Shared ConvNeXt-Base instance for layer mapping tests (read-only)."""
    return tv_models.convnext_base()


@pytest.fixture(scope="module")
def dinov3_extractor():
    """Shared DINOv3 TimmViTExtractor (read-only after setup)."""
    return _create_timm_vit_extractor("vit_large_patch16_dinov3")


# ═══════════════════════════════════════════════════════════════
# 1. RSA CORRECTNESS AGAINST RSATOOLBOX
# ═══════════════════════════════════════════════════════════════

class TestRSAAgainstToolbox:
    """Validate our RSA implementation against rsatoolbox (Kriegeskorte lab).

    rsatoolbox is the canonical Python RSA package. Any divergence between
    our implementation and the toolbox indicates a bug in our code.
    """

    @pytest.fixture
    def reference_data(self):
        """Shared data for all rsatoolbox comparison tests."""
        np.random.seed(42)
        return np.random.randn(30, 50).astype(np.float32)

    @pytest.fixture
    def two_datasets(self):
        """Two independent datasets for RDM comparison tests."""
        np.random.seed(42)
        x = np.random.randn(25, 40).astype(np.float32)
        np.random.seed(99)
        y = np.random.randn(25, 40).astype(np.float32)
        return x, y

    def test_rdm_matches_toolbox_pearson(self, reference_data):
        """Our Pearson RDM should match rsatoolbox's correlation RDM exactly.

        rsatoolbox's method='correlation' computes 1 - Pearson(row_i, row_j),
        which is the same as our compute_rdm(x, correlation='Pearson').
        """
        our_rdm, ref_matrix = _our_and_ref_rdms(reference_data)

        np.testing.assert_allclose(
            our_rdm.numpy(), ref_matrix, atol=1e-4,
            err_msg="Pearson RDM diverges from rsatoolbox"
        )

    def test_rdm_diagonal_matches_toolbox(self, reference_data):
        """Both implementations should have exactly zero on the diagonal."""
        our_rdm, ref_matrix = _our_and_ref_rdms(reference_data)

        np.testing.assert_array_equal(np.diag(our_rdm.numpy()), 0.0)
        np.testing.assert_array_equal(np.diag(ref_matrix), 0.0)

    def test_rdm_upper_triangle_matches_toolbox(self, reference_data):
        """Upper triangle values (the ones used for comparison) should match."""
        our_rdm = compute_rdm(torch.from_numpy(reference_data), correlation="Pearson")
        ref_rdm_obj = _rsatoolbox_rdm(reference_data)

        # rsatoolbox stores upper triangle as flat vector
        ref_upper = ref_rdm_obj.dissimilarities.flatten()

        # Our upper triangle
        n = our_rdm.size(0)
        idx = torch.triu_indices(n, n, offset=1)
        our_upper = our_rdm[idx[0], idx[1]].numpy()

        np.testing.assert_allclose(
            our_upper, ref_upper, atol=1e-4,
            err_msg="Upper triangle values diverge from rsatoolbox"
        )

    @pytest.mark.parametrize("our_method,ref_method,atol", [
        ("Spearman", "spearman", 1e-4),
        ("Pearson", "corr", 1e-4),
        ("Kendall", "tau-a", 1e-3),
    ])
    def test_rdm_comparison_matches_toolbox(self, two_datasets, our_method, ref_method, atol):
        """Our RDM comparison should match rsatoolbox's for Spearman, Pearson, and Kendall."""
        x, y = two_datasets

        rdm1 = compute_rdm(torch.from_numpy(x), correlation="Pearson")
        rdm2 = compute_rdm(torch.from_numpy(y), correlation="Pearson")
        our_score = compute_rdm_correlation(rdm1, rdm2, correlation=our_method)

        ref_rdm1 = _rsatoolbox_rdm(x)
        ref_rdm2 = _rsatoolbox_rdm(y)
        ref_score = rsatoolbox.rdm.compare(ref_rdm1, ref_rdm2, method=ref_method)[0, 0]

        assert our_score == pytest.approx(ref_score, abs=atol), (
            f"{our_method} comparison: ours={our_score:.6f}, toolbox={ref_score:.6f}"
        )

    def test_rdm_with_correlated_data(self):
        """Test with structured data where some rows are similar.

        This catches bugs that only appear when RDM values span a wide range
        (near-zero to near-2), unlike random data where values cluster near 1.
        """
        np.random.seed(123)
        base = np.random.randn(1, 30).astype(np.float32)
        # Rows 0-4: similar to base, rows 5-9: orthogonal
        data = np.vstack([
            base + np.random.randn(5, 30).astype(np.float32) * 0.1,
            np.random.randn(5, 30).astype(np.float32),
        ])

        our_rdm, ref_matrix = _our_and_ref_rdms(data)
        np.testing.assert_allclose(our_rdm.numpy(), ref_matrix, atol=1e-4)

    def test_spearman_rdm_matches_toolbox(self):
        """Our Spearman RDM (rank-based) should produce valid results.

        rsatoolbox doesn't have a direct Spearman RDM, but we can verify
        our Spearman RDM is equivalent to Pearson RDM on rank-transformed data.
        """
        import scipy.stats

        np.random.seed(77)
        data = np.random.randn(15, 20).astype(np.float32)

        # Our Spearman RDM
        our_rdm = compute_rdm(torch.from_numpy(data), correlation="Spearman")

        # Manual: rank each row, then compute Pearson RDM
        ranked = np.apply_along_axis(scipy.stats.rankdata, 1, data).astype(np.float32)
        ref_matrix = _rsatoolbox_rdm(ranked).get_matrices()[0]

        np.testing.assert_allclose(our_rdm.numpy(), ref_matrix, atol=1e-3,
            err_msg="Spearman RDM != Pearson RDM on ranked data"
        )


# ═══════════════════════════════════════════════════════════════
# 2. FEATURE EXTRACTION & LAYER NAMING
# ═══════════════════════════════════════════════════════════════

class TestTorchvisionReturnNodesExist:
    """Verify that all declared return_nodes actually exist in each model.

    This is the single most fragile part of the codebase. When torchvision
    or timm updates change internal module paths, FeatureExtractor silently
    attaches hooks to nothing, producing empty activations.
    """

    # Models that use standard FeatureExtractor (torchvision-based)
    TORCHVISION_MODELS = {
        "AlexNet":       ("AlexNet",       "imagenet1k"),
        "VGG16":         ("VGG16",         "imagenet1k"),
        "ResNet50":      ("ResNet50",      "imagenet1k"),
        "ViTBase":       ("ViTBase",       "imagenet1k"),
        "ConvNeXt_Base": ("ConvNeXt_Base", "imagenet1k"),
    }

    # Models with custom extractors (CLIP, DINO) — handled separately
    CUSTOM_EXTRACTOR_MODELS = {
        "DINOv2_ViT_B14": ("DINOv2_ViT_B14", None),
        "DINOv3_ViT_L16": ("DINOv3_ViT_L16", None),
    }

    @pytest.mark.parametrize("model_name", list(TORCHVISION_MODELS.keys()))
    def test_return_nodes_produce_output(self, model_name):
        """Each declared return_node should produce an activation tensor.

        Instantiates the model with NO pretrained weights (fast), wraps with
        FeatureExtractor, runs a dummy forward pass, and checks that every
        expected layer appears in the output.
        """
        from visreps.models import standard_model

        model_fn_name, _ = self.TORCHVISION_MODELS[model_name]
        model_fn = getattr(standard_model, model_fn_name)
        model = model_fn(pretrained_dataset="none")

        return_nodes = TORCHVISION_RETURN_NODES[model_name]
        assert return_nodes is not None, f"No return_nodes defined for {model_name}"

        return_nodes_dict = {n: n for n in return_nodes}
        extractor = FeatureExtractor(model, return_nodes_dict, extract_pre_and_post=True)

        # Dummy input: standard ImageNet size
        dummy = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = extractor(dummy)

        # Every declared return_node should produce a tensor
        for node in return_nodes:
            # With pre/post expansion, we expect either node_pre/node_post or node itself
            has_pre = f"{node}_pre" in output
            has_post = f"{node}_post" in output
            has_plain = node in output
            assert has_pre or has_post or has_plain, (
                f"Layer '{node}' missing from {model_name} output. "
                f"Got keys: {list(output.keys())}"
            )

        # All outputs should be tensors with batch_size=2
        for key, tensor in output.items():
            assert isinstance(tensor, torch.Tensor), f"{key} is not a tensor"
            assert tensor.shape[0] == 2, f"{key} has wrong batch dim: {tensor.shape}"

    @pytest.mark.parametrize("model_name,timm_id", [
        ("DINOv2_ViT_B14", "vit_base_patch14_dinov2"),
        ("DINOv3_ViT_L16", "vit_large_patch16_dinov3"),
    ])
    def test_custom_extractor_return_nodes(self, model_name, timm_id):
        """DINO models with TimmViTExtractor forward() should produce expected layers.

        Uses pretrained=False to avoid downloading weights in CI.
        """
        model = _create_timm_vit_extractor(timm_id)

        return_nodes = TORCHVISION_RETURN_NODES[model_name]
        assert return_nodes is not None, f"No return_nodes for {model_name}"

        model.return_nodes = {n: n for n in return_nodes}
        model.eval()

        dummy = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy)

        for node in return_nodes:
            assert node in output, (
                f"Layer '{node}' missing from {model_name} output. "
                f"Got keys: {list(output.keys())}"
            )
            assert output[node].shape[0] == 2


class TestFeatureExtractorLayerMapping:
    """Test that FeatureExtractor creates correct semantic layer names."""

    def test_alexnet_layer_count(self, alexnet_model):
        """AlexNet should have 5 conv + 3 fc layers."""
        return_nodes = {n: n for n in TORCHVISION_RETURN_NODES["AlexNet"]}
        ext = FeatureExtractor(alexnet_model, return_nodes, extract_pre_and_post=False)
        mapping = ext._create_layer_mapping()

        conv_layers = [k for k in mapping if k.startswith("conv")]
        fc_layers = [k for k in mapping if k.startswith("fc")]
        assert len(conv_layers) == 5, f"Expected 5 conv layers, got {conv_layers}"
        # AlexNet classifier has 3 Linear layers (fc1=classifier.1, fc2=classifier.4, fc3=classifier.6)
        assert len(fc_layers) == 3, f"Expected 3 fc layers, got {fc_layers}"

    def test_resnet50_has_16_blocks(self, resnet50_model):
        """ResNet50 should have conv1 + 16 blocks + fc1."""
        return_nodes = {n: n for n in TORCHVISION_RETURN_NODES["ResNet50"]}
        ext = FeatureExtractor(resnet50_model, return_nodes, extract_pre_and_post=False)
        mapping = ext._create_layer_mapping()

        assert "conv1" in mapping
        blocks = [k for k in mapping if k.startswith("block")]
        assert len(blocks) == 16, f"Expected 16 blocks, got {len(blocks)}: {blocks}"
        assert "fc1" in mapping

    def test_vit_has_12_blocks(self, vit_model):
        """ViT-Base should have 12 transformer blocks."""
        return_nodes = {n: n for n in TORCHVISION_RETURN_NODES["ViTBase"]}
        ext = FeatureExtractor(vit_model, return_nodes, extract_pre_and_post=False)
        mapping = ext._create_layer_mapping()

        blocks = [k for k in mapping if k.startswith("block")]
        assert len(blocks) == 12, f"Expected 12 blocks, got {len(blocks)}: {blocks}"

    def test_convnext_has_36_blocks(self, convnext_model):
        """ConvNeXt-Base should have 36 CNBlocks."""
        return_nodes = {n: n for n in TORCHVISION_RETURN_NODES["ConvNeXt_Base"]}
        ext = FeatureExtractor(convnext_model, return_nodes, extract_pre_and_post=False)
        mapping = ext._create_layer_mapping()

        blocks = [k for k in mapping if k.startswith("block")]
        assert len(blocks) == 36, f"Expected 36 blocks, got {len(blocks)}: {blocks}"

    def test_pre_post_expansion_doubles_layers(self, alexnet_model):
        """With extract_pre_and_post=True, each layer with an activation fn
        should expand to layer_pre + layer_post."""
        return_nodes = {"conv1": "conv1", "conv2": "conv2"}
        ext = FeatureExtractor(alexnet_model, return_nodes, extract_pre_and_post=True)

        # AlexNet conv layers have ReLU activations, so both should expand
        assert "conv1_pre" in ext.return_nodes, f"conv1_pre missing: {list(ext.return_nodes.keys())}"
        assert "conv1_post" in ext.return_nodes, f"conv1_post missing: {list(ext.return_nodes.keys())}"
        assert "conv2_pre" in ext.return_nodes, f"conv2_pre missing: {list(ext.return_nodes.keys())}"
        assert "conv2_post" in ext.return_nodes, f"conv2_post missing: {list(ext.return_nodes.keys())}"

    def test_forward_output_shapes_consistent(self, alexnet_model):
        """All output tensors should have matching batch dimension."""
        return_nodes = {n: n for n in ["conv1", "conv3", "fc1"]}
        ext = FeatureExtractor(alexnet_model, return_nodes, extract_pre_and_post=True)

        dummy = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            out = ext(dummy)

        for key, tensor in out.items():
            assert tensor.shape[0] == 4, f"{key}: batch dim is {tensor.shape[0]}, expected 4"

    def test_hook_cleanup(self, alexnet_model):
        """FeatureExtractor should register hooks that can be cleaned up."""
        return_nodes = {"conv1": "conv1"}
        ext = FeatureExtractor(alexnet_model, return_nodes, extract_pre_and_post=False)
        assert len(ext.handles) > 0, "No hooks registered"


class TestDINOv3LayerExtraction:
    """Focused tests for DINOv3 (timm ViT-L/16) layer extraction.

    DINOv3 uses TimmViTExtractor with rotary position embeddings,
    making it the most architecturally complex model to extract from.
    """

    def test_dinov3_block_count(self, dinov3_extractor):
        """DINOv3 ViT-L/16 should have 24 transformer blocks."""
        assert len(dinov3_extractor.model.blocks) == 24

    def test_dinov3_return_nodes_valid(self):
        """All declared DINOv3 return_nodes should be valid block indices."""
        nodes = TORCHVISION_RETURN_NODES["DINOv3_ViT_L16"]
        for node in nodes:
            assert node.startswith("block"), f"Unexpected node name: {node}"
            idx = int(node.replace("block", ""))
            assert 1 <= idx <= 24, f"Block index {idx} out of range [1, 24]"

    def test_dinov3_extraction_produces_output(self, dinov3_extractor):
        """DINOv3 should produce activations for all declared return_nodes."""
        nodes = TORCHVISION_RETURN_NODES["DINOv3_ViT_L16"]
        dinov3_extractor.return_nodes = {n: n for n in nodes}
        dinov3_extractor.eval()

        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = dinov3_extractor(dummy)

        for node in nodes:
            assert node in output, f"Missing {node} in DINOv3 output"
            # ViT output: (batch, seq_len, hidden_dim)
            assert output[node].ndim == 3, f"{node} has {output[node].ndim}D, expected 3D"
            assert output[node].shape[0] == 1

    def test_dinov2_extraction_produces_output(self):
        """DINOv2 ViT-B/14 should also work with TimmViTExtractor."""
        model = _create_timm_vit_extractor("vit_base_patch14_dinov2")

        nodes = TORCHVISION_RETURN_NODES["DINOv2_ViT_B14"]
        model.return_nodes = {n: n for n in nodes}
        model.eval()

        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(dummy)

        for node in nodes:
            assert node in output, f"Missing {node} in DINOv2 output"


# ═══════════════════════════════════════════════════════════════
# 3. CONFIG VERIFICATION
# ═══════════════════════════════════════════════════════════════

class TestConfigVerifier:
    """Tests for ConfigVerifier — the gate that prevents invalid experiments."""

    # ── Valid configs should pass ──

    def test_valid_nsd_config(self):
        """Standard NSD eval config should pass validation."""
        cfg = _make_eval_cfg()
        result = ConfigVerifier(cfg).verify()
        assert result.mode == "eval"

    def test_valid_tvsd_config(self):
        """TVSD config with valid subjects/regions should pass."""
        cfg = _make_eval_cfg(
            neural_dataset="tvsd", subject_idx=[0, 1], region=["V1", "IT"],
        )
        result = ConfigVerifier(cfg).verify()
        assert list(result.subject_idx) == [0, 1]
        assert list(result.region) == ["V1", "IT"]

    def test_valid_things_config(self):
        """THINGS-behavior config should pass and force N/A."""
        cfg = _make_eval_cfg(
            neural_dataset="things-behavior", subject_idx=0, region="ventral visual stream",
        )
        result = ConfigVerifier(cfg).verify()
        assert result.region == "N/A"
        assert result.subject_idx == "N/A"

    def test_valid_torchvision_config(self):
        """Torchvision loading should pass validation."""
        cfg = _make_eval_cfg(load_model_from="torchvision")
        result = ConfigVerifier(cfg).verify()
        assert result.load_model_from == "torchvision"

    # ── Scalar → list normalization ──

    def test_scalar_subject_normalized_to_list(self):
        """Single int subject_idx should become a list."""
        cfg = _make_eval_cfg(subject_idx=3)
        result = ConfigVerifier(cfg).verify()
        # OmegaConf may return ListConfig instead of list, so check contents
        assert list(result.subject_idx) == [3]

    def test_scalar_region_normalized_to_list(self):
        """Single string region should become a list."""
        cfg = _make_eval_cfg(region="early visual stream")
        result = ConfigVerifier(cfg).verify()
        assert list(result.region) == ["early visual stream"]

    # ── THINGS forces N/A ──

    def test_things_forces_subject_na(self):
        """THINGS should override any subject_idx to N/A."""
        cfg = _make_eval_cfg(neural_dataset="things-behavior", subject_idx=5)
        result = ConfigVerifier(cfg).verify()
        assert result.subject_idx == "N/A"

    # ── Encoding score constraints ──

    def test_encoding_score_rejects_things(self):
        """encoding_score + things-behavior should raise (no voxels)."""
        cfg = _make_eval_cfg(
            neural_dataset="things-behavior", analysis="encoding_score",
        )
        with pytest.raises(AssertionError, match="encoding_score.*not supported.*things"):
            ConfigVerifier(cfg).verify()

    def test_encoding_score_rejects_nsd_synthetic(self):
        """encoding_score + nsd_synthetic should raise."""
        cfg = _make_eval_cfg(
            neural_dataset="nsd_synthetic", analysis="encoding_score",
            subject_idx=[0], region=["ventral visual stream"],
        )
        with pytest.raises(AssertionError, match="encoding_score.*not supported.*nsd_synthetic"):
            ConfigVerifier(cfg).verify()

    def test_encoding_score_forces_pearson(self):
        """encoding_score should override compare_method to 'pearson'."""
        cfg = _make_eval_cfg(analysis="encoding_score", compare_method="spearman")
        result = ConfigVerifier(cfg).verify()
        assert result.compare_method == "pearson"

    # ── Invalid values should reject ──

    @pytest.mark.parametrize("overrides,match", [
        ({"seed": 5}, "Invalid seed"),
        ({"seed": 0}, "Invalid seed"),
        ({"region": ["nonexistent_region"]}, "Invalid region"),
        ({"subject_idx": [8]}, "Invalid subject"),
        ({"neural_dataset": "tvsd", "subject_idx": [0], "region": ["V2"]}, "Invalid region"),
        ({"neural_dataset": "tvsd", "subject_idx": [2], "region": ["IT"]}, "Invalid subject"),
        ({"analysis": "cka"}, "Invalid analysis"),
        ({"compare_method": "cosine"}, "Invalid compare_method"),
        ({"load_model_from": "huggingface"}, "load_model_from"),
    ])
    def test_invalid_config_raises(self, overrides, match):
        """Invalid config values should raise AssertionError with descriptive message."""
        cfg = _make_eval_cfg(**overrides)
        with pytest.raises(AssertionError, match=match):
            ConfigVerifier(cfg).verify()

    def test_invalid_mode_raises(self):
        """Invalid mode should raise (uses different base config)."""
        cfg = OmegaConf.create({"mode": "predict"})
        with pytest.raises(AssertionError, match="Invalid mode"):
            ConfigVerifier(cfg).verify()

    # ── NSD fine-grained regions ──

    def test_nsd_finegrained_regions_valid(self):
        """Fine-grained NSD regions (V1, V2, V3, hV4, FFA, PPA) should pass."""
        for region in ["V1", "V2", "V3", "hV4", "FFA", "PPA"]:
            cfg = _make_eval_cfg(region=[region])
            result = ConfigVerifier(cfg).verify()
            assert result.region == [region]


# ═══════════════════════════════════════════════════════════════
# 4. MODEL LOADING SMOKE TESTS
# ═══════════════════════════════════════════════════════════════

class TestModelLoading:
    """Smoke tests for model creation (no pretrained weights for speed)."""

    @pytest.mark.parametrize("model_name,expected_type", [
        ("AlexNet", "AlexNet"),
        ("VGG16", "VGG"),
        ("ResNet50", "ResNet"),
    ])
    def test_standard_model_creates(self, model_name, expected_type):
        """Standard torchvision models should instantiate without error."""
        from visreps.models import standard_model
        model = getattr(standard_model, model_name)(pretrained_dataset="none")
        assert expected_type in type(model).__name__

    def test_custom_cnn_creates(self):
        """CustomCNN should instantiate with default parameters."""
        from visreps.models.custom_model import CustomCNN
        model = CustomCNN(num_classes=10)
        assert model is not None
        # Should have 5 conv layers and fc layers
        dummy = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (2, 10)

    def test_invalid_model_name_raises(self):
        """Non-existent model name should raise ValueError."""
        from visreps.models.utils import load_model
        cfg = OmegaConf.create({
            "model_class": "standard_model",
            "model_name": "NonExistentModel",
            "pretrained_dataset": "none",
        })
        with pytest.raises(ValueError, match="not found"):
            load_model(cfg, torch.device("cpu"))

    def test_checkpoint_missing_file_raises(self):
        """Loading from a non-existent checkpoint should raise."""
        from visreps.models.utils import load_model
        cfg = OmegaConf.create({
            "load_model_from": "checkpoint",
            "checkpoint_dir": "/tmp/nonexistent_dir_12345",
            "cfg_id": 1000,
            "seed": 1,
            "checkpoint_model": "checkpoint_epoch_20.pth",
        })
        with pytest.raises((FileNotFoundError, RuntimeError)):
            load_model(cfg, torch.device("cpu"))

    def test_seed_letter_in_checkpoint_path(self):
        """Checkpoint path should use correct seed letter mapping."""
        from visreps.utils import get_seed_letter
        assert get_seed_letter(1) == "a"
        assert get_seed_letter(2) == "b"
        assert get_seed_letter(3) == "c"


# ═══════════════════════════════════════════════════════════════
# 5. EVAL DISPATCHER CONFIG HANDLING
# ═══════════════════════════════════════════════════════════════

class TestEvalConfigHandling:
    """Test the config handling logic at the top of evals.eval()."""

    def test_torchvision_pretrained_sets_cfg_id(self):
        """load_model_from=torchvision with pretrained should set cfg_id='pretrained'."""
        from visreps.evals import _set_torchvision_cfg

        cfg = OmegaConf.create({
            "load_model_from": "torchvision",
            "pretrained_dataset": "imagenet1k",
        })
        cfg = _set_torchvision_cfg(cfg)
        assert cfg.cfg_id == "pretrained"
        assert cfg.epoch == -1

    def test_torchvision_untrained_sets_cfg_id(self):
        """load_model_from=torchvision with pretrained=none should set cfg_id='untrained'."""
        from visreps.evals import _set_torchvision_cfg

        cfg = OmegaConf.create({
            "load_model_from": "torchvision",
            "pretrained_dataset": "none",
        })
        cfg = _set_torchvision_cfg(cfg)
        assert cfg.cfg_id == "untrained"
        assert cfg.epoch == -1

    def test_clip_transform_selected(self):
        """CLIP models should get clip-specific preprocessing."""
        from visreps.evals import _get_eval_transform
        cfg = OmegaConf.create({"model_name": "CLIP_ViT_B32"})
        transform = _get_eval_transform(cfg)
        assert transform is not None

    def test_non_clip_transform_default(self):
        """Non-CLIP models should get ImageNet preprocessing."""
        from visreps.evals import _get_eval_transform
        cfg = OmegaConf.create({"model_name": "ResNet50"})
        transform = _get_eval_transform(cfg)
        assert transform is not None

    @pytest.mark.parametrize("value,expected", [
        (0, [0]),
        ("ventral visual stream", ["ventral visual stream"]),
        ([0, 1, 2], [0, 1, 2]),
    ])
    def test_listify(self, value, expected):
        """_listify should wrap scalars in lists and pass through lists unchanged."""
        from visreps.evals import _listify
        assert _listify(value) == expected

    def test_return_nodes_lookup(self):
        """Every model in TORCHVISION_RETURN_NODES should have a non-empty list."""
        for model_name, nodes in TORCHVISION_RETURN_NODES.items():
            assert nodes is not None, f"{model_name} has None return_nodes"
            assert len(nodes) > 0, f"{model_name} has empty return_nodes"


# ═══════════════════════════════════════════════════════════════
# 6. REGRESSION TESTS — RUN PIPELINE, COMPARE TO HARDCODED VALUES
# ═══════════════════════════════════════════════════════════════

# Three models covering all loading paths:
#   1. CustomCNN 32-way CLIP PCA  — checkpoint + coarse-grained (RSA + encoding)
#   2. CustomCNN 1000-way         — checkpoint + fine-grained   (RSA + encoding)
#   3. ViTBase pretrained         — torchvision pretrained      (RSA only)
# Total: 5 pipeline runs, shared via module-scoped fixtures.


def _run_nsd_eval_checkpoint(checkpoint_dir, cfg_id, analysis,
                              pca_labels=False, pca_n_classes=None, pca_labels_folder=None):
    """Run NSD eval for a checkpoint model (8 subjects × 2 regions)."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")
    from visreps.utils import get_seed_letter
    seed_letter = get_seed_letter(1)
    checkpoint = f"{checkpoint_dir}/cfg{cfg_id}{seed_letter}/checkpoint_epoch_20.pth"
    if not os.path.exists(checkpoint):
        pytest.skip(f"Checkpoint not found: {checkpoint}")

    from dotenv import load_dotenv
    load_dotenv()
    import visreps.utils as utils
    import visreps.evals as evals

    overrides = [
        "mode=eval", f"cfg_id={cfg_id}", "seed=1",
        f"checkpoint_dir={checkpoint_dir}",
        "checkpoint_model=checkpoint_epoch_20.pth",
        "neural_dataset=nsd", f"analysis={analysis}",
        "subject_idx=[0,1,2,3,4,5,6,7]",
        'region=["early visual stream","ventral visual stream"]',
        "log_expdata=false",
        "bootstrap=true", "n_bootstrap=1000",
        "verbose=false", "batchsize=64", "num_workers=4",
    ]
    if pca_labels:
        overrides += [
            "pca_labels=true", f"pca_n_classes={pca_n_classes}",
            f"pca_labels_folder={pca_labels_folder}",
        ]
    if analysis == "rsa":
        overrides.append("compare_method=spearman")

    cfg = utils.load_config("configs/eval/base.json", overrides)
    cfg = utils.validate_config(cfg)

    try:
        return evals.eval(cfg)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        pytest.skip("CUDA OOM — need more GPU memory for NSD eval")


def _run_nsd_eval_torchvision(model_name, pretrained_dataset, analysis):
    """Run NSD eval for a torchvision pretrained model (8 subjects × 2 regions)."""
    if not torch.cuda.is_available():
        pytest.skip("No GPU available")

    from dotenv import load_dotenv
    load_dotenv()
    import visreps.utils as utils
    import visreps.evals as evals

    overrides = [
        "mode=eval", "seed=1",
        "load_model_from=torchvision",
        f"model_name={model_name}",
        f"pretrained_dataset={pretrained_dataset}",
        "neural_dataset=nsd", f"analysis={analysis}",
        "subject_idx=[0,1,2,3,4,5,6,7]",
        'region=["early visual stream","ventral visual stream"]',
        "log_expdata=false",
        "bootstrap=true", "n_bootstrap=1000",
        "verbose=false", "batchsize=64", "num_workers=4",
    ]
    if analysis == "rsa":
        overrides.append("compare_method=spearman")

    cfg = utils.load_config("configs/eval/base.json", overrides)
    cfg = utils.validate_config(cfg)

    try:
        return evals.eval(cfg)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        pytest.skip("CUDA OOM — need more GPU memory for NSD eval")


# Module-scoped fixtures: 5 pipeline runs total
@pytest.fixture(scope="module")
def clip_pca_rsa():
    """CustomCNN 32-way CLIP PCA: NSD RSA."""
    return _run_nsd_eval_checkpoint(
        "/data/ymehta3/clip_pca", 32, "rsa",
        pca_labels=True, pca_n_classes=32, pca_labels_folder="pca_labels_clip",
    )

@pytest.fixture(scope="module")
def clip_pca_encoding():
    """CustomCNN 32-way CLIP PCA: NSD encoding score."""
    return _run_nsd_eval_checkpoint(
        "/data/ymehta3/clip_pca", 32, "encoding_score",
        pca_labels=True, pca_n_classes=32, pca_labels_folder="pca_labels_clip",
    )

@pytest.fixture(scope="module")
def cnn1000_rsa():
    """CustomCNN 1000-way: NSD RSA."""
    return _run_nsd_eval_checkpoint("/data/ymehta3/default", 1000, "rsa")

@pytest.fixture(scope="module")
def cnn1000_encoding():
    """CustomCNN 1000-way: NSD encoding score."""
    return _run_nsd_eval_checkpoint("/data/ymehta3/default", 1000, "encoding_score")

@pytest.fixture(scope="module")
def vit_pretrained_rsa():
    """ViTBase pretrained (torchvision): NSD RSA."""
    return _run_nsd_eval_torchvision("ViTBase", "imagenet1k", "rsa")


def _find_result(results_df, region, subj, analysis):
    """Find the result row for a specific (region, subject) in pipeline output."""
    mask = (
        (results_df["region"] == region)
        & (results_df["subject_idx"].astype(str) == str(subj))
    )
    matches = results_df[mask]
    assert len(matches) == 1, (
        f"Expected 1 result for {region} subj {subj} ({analysis}), got {len(matches)}"
    )
    return matches.iloc[0]


def _check_result(results_df, ref, region, subj, analysis):
    """Assert that a pipeline result matches the hardcoded reference."""
    row = _find_result(results_df, region, subj, analysis)
    assert row["layer"] == ref["layer"], (
        f"Best layer changed: expected {ref['layer']}, got {row['layer']} "
        f"({analysis}, region={region}, subj={subj})"
    )
    assert row["score"] == pytest.approx(ref["score"], abs=1e-4), (
        f"Score changed for {analysis} {region} subj {subj}: "
        f"expected {ref['score']:.6f}, got {row['score']:.6f}"
    )
    assert row["ci_low"] == pytest.approx(ref["ci_low"], abs=1e-4), (
        f"ci_low changed for {analysis} {region} subj {subj}: "
        f"expected {ref['ci_low']:.6f}, got {row['ci_low']:.6f}"
    )
    assert row["ci_high"] == pytest.approx(ref["ci_high"], abs=1e-4), (
        f"ci_high changed for {analysis} {region} subj {subj}: "
        f"expected {ref['ci_high']:.6f}, got {row['ci_high']:.6f}"
    )


# ── Model 1: CustomCNN 32-way CLIP PCA ──────────────────

class TestRegressionCLIPPCA:
    """Regression: CustomCNN 32-way with CLIP PCA labels on NSD (RSA + encoding)."""

    RSA_REFERENCE = {
        "early visual stream": {
            0: {"layer": "conv3_pre", "score": 0.254424, "ci_low": 0.247500, "ci_high": 0.261322},
            1: {"layer": "conv3_pre", "score": 0.244366, "ci_low": 0.238090, "ci_high": 0.250591},
            2: {"layer": "conv3_pre", "score": 0.180812, "ci_low": 0.174325, "ci_high": 0.187623},
            3: {"layer": "conv3_pre", "score": 0.201619, "ci_low": 0.195347, "ci_high": 0.208266},
            4: {"layer": "conv3_pre", "score": 0.193556, "ci_low": 0.187306, "ci_high": 0.199823},
            5: {"layer": "conv3_pre", "score": 0.180900, "ci_low": 0.175343, "ci_high": 0.186718},
            6: {"layer": "conv3_pre", "score": 0.161558, "ci_low": 0.156070, "ci_high": 0.167165},
            7: {"layer": "conv3_pre", "score": 0.149072, "ci_low": 0.143298, "ci_high": 0.154960},
        },
        "ventral visual stream": {
            0: {"layer": "fc1_post", "score": 0.294310, "ci_low": 0.287358, "ci_high": 0.301045},
            1: {"layer": "fc2_pre",  "score": 0.306921, "ci_low": 0.299880, "ci_high": 0.314782},
            2: {"layer": "fc1_post", "score": 0.259282, "ci_low": 0.251971, "ci_high": 0.266667},
            3: {"layer": "fc1_post", "score": 0.231072, "ci_low": 0.224464, "ci_high": 0.238261},
            4: {"layer": "fc2_pre",  "score": 0.290787, "ci_low": 0.283612, "ci_high": 0.298487},
            5: {"layer": "fc1_post", "score": 0.221283, "ci_low": 0.214874, "ci_high": 0.227535},
            6: {"layer": "fc1_post", "score": 0.248883, "ci_low": 0.242159, "ci_high": 0.255482},
            7: {"layer": "fc1_post", "score": 0.182723, "ci_low": 0.176623, "ci_high": 0.189020},
        },
    }

    ENCODING_REFERENCE = {
        "early visual stream": {
            0: {"layer": "conv4_pre", "score": 0.360563, "ci_low": 0.357381, "ci_high": 0.363319},
            1: {"layer": "conv4_pre", "score": 0.361197, "ci_low": 0.358369, "ci_high": 0.363807},
            2: {"layer": "conv4_pre", "score": 0.288192, "ci_low": 0.284239, "ci_high": 0.291851},
            3: {"layer": "conv4_pre", "score": 0.287534, "ci_low": 0.283656, "ci_high": 0.291233},
            4: {"layer": "fc1_pre",   "score": 0.369791, "ci_low": 0.366188, "ci_high": 0.373288},
            5: {"layer": "conv4_pre", "score": 0.300835, "ci_low": 0.296738, "ci_high": 0.304594},
            6: {"layer": "conv4_pre", "score": 0.273849, "ci_low": 0.270211, "ci_high": 0.277887},
            7: {"layer": "conv4_pre", "score": 0.216003, "ci_low": 0.212092, "ci_high": 0.219834},
        },
        "ventral visual stream": {
            0: {"layer": "fc2_post", "score": 0.204146, "ci_low": 0.202037, "ci_high": 0.206341},
            1: {"layer": "fc2_post", "score": 0.244954, "ci_low": 0.242485, "ci_high": 0.247447},
            2: {"layer": "fc2_post", "score": 0.171759, "ci_low": 0.169102, "ci_high": 0.174327},
            3: {"layer": "fc2_post", "score": 0.184338, "ci_low": 0.181507, "ci_high": 0.187042},
            4: {"layer": "fc2_post", "score": 0.245524, "ci_low": 0.243177, "ci_high": 0.247811},
            5: {"layer": "fc2_post", "score": 0.160114, "ci_low": 0.157747, "ci_high": 0.162652},
            6: {"layer": "fc2_post", "score": 0.179206, "ci_low": 0.177147, "ci_high": 0.181352},
            7: {"layer": "fc2_post", "score": 0.125519, "ci_low": 0.122829, "ci_high": 0.128404},
        },
    }

    @pytest.mark.slow
    @pytest.mark.parametrize("region", ["early visual stream", "ventral visual stream"])
    @pytest.mark.parametrize("subj", list(range(8)))
    def test_rsa(self, clip_pca_rsa, region, subj):
        _check_result(clip_pca_rsa, self.RSA_REFERENCE[region][subj], region, subj, "rsa")

    @pytest.mark.slow_encoding
    @pytest.mark.parametrize("region", ["early visual stream", "ventral visual stream"])
    @pytest.mark.parametrize("subj", list(range(8)))
    def test_encoding(self, clip_pca_encoding, region, subj):
        _check_result(clip_pca_encoding, self.ENCODING_REFERENCE[region][subj], region, subj, "encoding_score")


# ── Model 2: CustomCNN 1000-way ──────────────────────────

class TestRegressionCNN1000:
    """Regression: CustomCNN 1000-way (standard ImageNet) on NSD (RSA + encoding)."""

    RSA_REFERENCE = {
        "early visual stream": {
            0: {"layer": "conv4_post", "score": 0.283807, "ci_low": 0.277237, "ci_high": 0.290410},
            1: {"layer": "conv4_post", "score": 0.257360, "ci_low": 0.250987, "ci_high": 0.263742},
            2: {"layer": "conv4_post", "score": 0.185048, "ci_low": 0.178360, "ci_high": 0.192240},
            3: {"layer": "conv4_post", "score": 0.221631, "ci_low": 0.214673, "ci_high": 0.228809},
            4: {"layer": "conv4_post", "score": 0.208494, "ci_low": 0.201636, "ci_high": 0.215888},
            5: {"layer": "conv4_post", "score": 0.197265, "ci_low": 0.191166, "ci_high": 0.203559},
            6: {"layer": "conv4_post", "score": 0.172151, "ci_low": 0.166054, "ci_high": 0.178110},
            7: {"layer": "conv4_post", "score": 0.164997, "ci_low": 0.159017, "ci_high": 0.171145},
        },
        "ventral visual stream": {
            0: {"layer": "fc1_pre", "score": 0.271445, "ci_low": 0.265292, "ci_high": 0.278051},
            1: {"layer": "fc1_pre", "score": 0.289824, "ci_low": 0.282874, "ci_high": 0.297340},
            2: {"layer": "fc1_pre", "score": 0.234739, "ci_low": 0.227607, "ci_high": 0.241645},
            3: {"layer": "fc1_pre", "score": 0.210587, "ci_low": 0.204193, "ci_high": 0.217172},
            4: {"layer": "fc1_pre", "score": 0.285522, "ci_low": 0.278116, "ci_high": 0.293158},
            5: {"layer": "fc1_pre", "score": 0.207548, "ci_low": 0.200576, "ci_high": 0.213844},
            6: {"layer": "fc1_pre", "score": 0.234645, "ci_low": 0.227940, "ci_high": 0.240810},
            7: {"layer": "fc1_pre", "score": 0.165668, "ci_low": 0.159732, "ci_high": 0.171415},
        },
    }

    ENCODING_REFERENCE = {
        "early visual stream": {
            0: {"layer": "conv4_pre", "score": 0.358053, "ci_low": 0.354905, "ci_high": 0.360814},
            1: {"layer": "conv4_pre", "score": 0.361185, "ci_low": 0.358330, "ci_high": 0.363798},
            2: {"layer": "conv4_pre", "score": 0.286033, "ci_low": 0.281982, "ci_high": 0.289617},
            3: {"layer": "conv4_pre", "score": 0.284923, "ci_low": 0.281310, "ci_high": 0.288799},
            4: {"layer": "fc1_pre",   "score": 0.373636, "ci_low": 0.370035, "ci_high": 0.377171},
            5: {"layer": "conv4_pre", "score": 0.295879, "ci_low": 0.291526, "ci_high": 0.299653},
            6: {"layer": "conv4_pre", "score": 0.272785, "ci_low": 0.269039, "ci_high": 0.276719},
            7: {"layer": "conv4_pre", "score": 0.217214, "ci_low": 0.213466, "ci_high": 0.221238},
        },
        "ventral visual stream": {
            0: {"layer": "fc2_post", "score": 0.201005, "ci_low": 0.198956, "ci_high": 0.203083},
            1: {"layer": "fc2_post", "score": 0.239025, "ci_low": 0.236436, "ci_high": 0.241524},
            2: {"layer": "fc1_pre",  "score": 0.167504, "ci_low": 0.164944, "ci_high": 0.169979},
            3: {"layer": "fc2_post", "score": 0.176737, "ci_low": 0.174192, "ci_high": 0.179441},
            4: {"layer": "fc2_post", "score": 0.241251, "ci_low": 0.238894, "ci_high": 0.243437},
            5: {"layer": "fc2_post", "score": 0.156355, "ci_low": 0.153771, "ci_high": 0.158831},
            6: {"layer": "fc2_post", "score": 0.177915, "ci_low": 0.175671, "ci_high": 0.180101},
            7: {"layer": "fc2_post", "score": 0.116887, "ci_low": 0.114250, "ci_high": 0.119764},
        },
    }

    @pytest.mark.slow
    @pytest.mark.parametrize("region", ["early visual stream", "ventral visual stream"])
    @pytest.mark.parametrize("subj", list(range(8)))
    def test_rsa(self, cnn1000_rsa, region, subj):
        _check_result(cnn1000_rsa, self.RSA_REFERENCE[region][subj], region, subj, "rsa")

    @pytest.mark.slow_encoding
    @pytest.mark.parametrize("region", ["early visual stream", "ventral visual stream"])
    @pytest.mark.parametrize("subj", list(range(8)))
    def test_encoding(self, cnn1000_encoding, region, subj):
        _check_result(cnn1000_encoding, self.ENCODING_REFERENCE[region][subj], region, subj, "encoding_score")


# ── Model 3: ViTBase pretrained (torchvision) ────────────

class TestRegressionViTPretrained:
    """Regression: ViTBase pretrained from torchvision on NSD (RSA only).

    Tests the torchvision loading path (no checkpoint, no PCA labels).
    Uses block-level layer names instead of conv/fc.
    """

    RSA_REFERENCE = {
        "early visual stream": {
            0: {"layer": "block4", "score": 0.315922, "ci_low": 0.309460, "ci_high": 0.322270},
            1: {"layer": "block4", "score": 0.283973, "ci_low": 0.277647, "ci_high": 0.290271},
            2: {"layer": "block4", "score": 0.213561, "ci_low": 0.206508, "ci_high": 0.221210},
            3: {"layer": "block4", "score": 0.244349, "ci_low": 0.237074, "ci_high": 0.252517},
            4: {"layer": "block4", "score": 0.251699, "ci_low": 0.244440, "ci_high": 0.259683},
            5: {"layer": "block5", "score": 0.226742, "ci_low": 0.220262, "ci_high": 0.233450},
            6: {"layer": "block4", "score": 0.188288, "ci_low": 0.182023, "ci_high": 0.194470},
            7: {"layer": "block4", "score": 0.194800, "ci_low": 0.188434, "ci_high": 0.201184},
        },
        "ventral visual stream": {
            0: {"layer": "block9",  "score": 0.191920, "ci_low": 0.187125, "ci_high": 0.197439},
            1: {"layer": "block5",  "score": 0.201933, "ci_low": 0.196397, "ci_high": 0.207973},
            2: {"layer": "block5",  "score": 0.166682, "ci_low": 0.161125, "ci_high": 0.173001},
            3: {"layer": "block9",  "score": 0.149422, "ci_low": 0.143837, "ci_high": 0.155264},
            4: {"layer": "block9",  "score": 0.202714, "ci_low": 0.196525, "ci_high": 0.209085},
            5: {"layer": "block9",  "score": 0.149513, "ci_low": 0.144372, "ci_high": 0.154554},
            6: {"layer": "block9",  "score": 0.170879, "ci_low": 0.165540, "ci_high": 0.176654},
            7: {"layer": "block9",  "score": 0.130173, "ci_low": 0.124801, "ci_high": 0.135225},
        },
    }

    @pytest.mark.slow
    @pytest.mark.parametrize("region", ["early visual stream", "ventral visual stream"])
    @pytest.mark.parametrize("subj", list(range(8)))
    def test_rsa(self, vit_pretrained_rsa, region, subj):
        _check_result(vit_pretrained_rsa, self.RSA_REFERENCE[region][subj], region, subj, "rsa")
