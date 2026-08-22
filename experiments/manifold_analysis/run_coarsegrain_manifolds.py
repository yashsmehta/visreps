"""Manifold analysis for CLIP-PCA coarse-grained CustomCNN checkpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from experiments.manifold_analysis.manifold_capacity import manifold_capacity
from experiments.manifold_analysis.manifold_snr import manifold_snr
from experiments.manifold_analysis.run_alexnet_manifolds import sample_images
from visreps.dataloaders.obj_cls import get_transform
from visreps.models import custom_model


CONDITIONS = (2, 4, 8, 16, 32, 64)
EXTENDED_CONDITIONS = (128, 256, 512)
SEEDS = ((1, "a"), (2, "b"), (3, "c"))
LAYERS = ("conv3_post", "fc2_post")
COARSE_CHECKPOINT_ROOT = Path("/data/ymehta3/clip_pca")
FINE_CHECKPOINT_ROOT = Path("/data/ymehta3/default")


def checkpoint_path(n_classes: int, seed_letter: str) -> Path:
    """Resolve coarse-label and conventional ImageNet checkpoint locations."""
    root = FINE_CHECKPOINT_ROOT if n_classes == 1000 else COARSE_CHECKPOINT_ROOT
    return root / f"cfg{n_classes}{seed_letter}" / "checkpoint_epoch_20.pth"


def load_model(n_classes: int, seed_letter: str, device: torch.device):
    """Load one serialized CustomCNN checkpoint."""
    sys.modules["visreps.models.custom_cnn"] = custom_model
    path = checkpoint_path(n_classes, seed_letter)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return checkpoint["model"].eval().to(device), path


def make_projection(
    n_features: int, *, seed: int, device: torch.device
) -> torch.Tensor | None:
    """Create the fixed Gaussian projection used for every model at one layer."""
    if n_features <= 2048:
        return None
    # Generate on CPU so CUDA and CPU runs use exactly the same matrix.
    generator = torch.Generator().manual_seed(seed)
    projection = torch.randn(n_features, 2048, generator=generator) / np.sqrt(2048)
    return projection.to(device)


@torch.inference_mode()
def extract_manifolds(
    model,
    image_paths: list[list[str]],
    *,
    batch_size: int,
    device: torch.device,
    projections: dict[str, torch.Tensor],
) -> dict[str, np.ndarray]:
    """Extract and project post-ReLU Conv3 and FC2 activations."""
    transform = get_transform(data_augment=False)
    flat_paths = [path for category in image_paths for path in category]
    layer_values = {layer: [] for layer in LAYERS}

    for start in range(0, len(flat_paths), batch_size):
        images = [
            transform(Image.open(path).convert("RGB"))
            for path in flat_paths[start : start + batch_size]
        ]
        x = torch.stack(images).to(device)

        x = model.features[:11](x)  # through Conv3 BatchNorm + ReLU
        layer_values["conv3_post"].append(
            (x.flatten(1) @ projections["conv3_post"]).cpu()
        )

        x = model.features[11:](x)
        x = model.adaptive_pool(x).flatten(1)
        fc2 = model.classifier[:8](x)  # through FC2 BatchNorm + ReLU
        layer_values["fc2_post"].append(
            (fc2 @ projections["fc2_post"]).cpu()
        )

    n_categories, n_images = len(image_paths), len(image_paths[0])
    return {
        layer: torch.cat(values).reshape(n_categories, n_images, 2048).permute(0, 2, 1).numpy()
        for layer, values in layer_values.items()
    }


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet-dir", type=Path, default=Path("/data/shared/datasets/imagenet"))
    parser.add_argument("--n-categories", type=int, default=100)
    parser.add_argument("--images-per-category", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-gpu-memory-gb", type=float, default=10.0)
    parser.add_argument("--n-probes", type=int, default=200)
    parser.add_argument("--panel-seed", type=int, default=2026)
    parser.add_argument("--metrics", choices=("snr", "capacity", "both"), default="both")
    parser.add_argument("--output-dir", type=Path, default=Path("experiments/manifold_analysis/coarsegrain_results"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel_path = args.output_dir / "image_panel.json"
    if panel_path.exists():
        panel = json.loads(panel_path.read_text())
        categories, image_paths = panel["categories"], panel["image_paths"]
    else:
        categories, image_paths = sample_images(
            args.imagenet_dir,
            args.n_categories,
            args.images_per_category,
            args.panel_seed,
        )
        save_json(panel_path, {"categories": categories, "image_paths": image_paths})

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested, but PyTorch cannot access CUDA")
    device = torch.device(
        "cuda:0" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
        else "cpu"
    )
    if device.type == "cuda":
        total_memory = torch.cuda.get_device_properties(device).total_memory
        memory_limit = min(args.max_gpu_memory_gb * 1024**3, total_memory)
        torch.cuda.set_per_process_memory_fraction(memory_limit / total_memory, device)
        torch.set_float32_matmul_precision("high")
        print(
            f"Using device: {device} "
            f"(PyTorch memory limit: {memory_limit / 1024**3:.1f} GB)"
        )
    else:
        print(f"Using device: {device}")

    # These large matrices are shared by all 18 models and created only once.
    projections = {
        "conv3_post": make_projection(384 * 13 * 13, seed=101, device=device),
        "fc2_post": make_projection(4096, seed=102, device=device),
    }
    results_path = args.output_dir / "metrics.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    # The original 2--64 sweep and 1,000-way baseline have three seeds. The
    # extended 128/256/512 CLIP-PCA checkpoints currently exist for seed a.
    jobs = [
        *[(n, seed, letter) for n in CONDITIONS for seed, letter in SEEDS],
        *[(n, 1, "a") for n in EXTENDED_CONDITIONS],
        *[(1000, seed, letter) for seed, letter in SEEDS],
    ]
    for n_classes, seed, seed_letter in tqdm(jobs, desc="models"):
        model_key = f"cfg{n_classes}{seed_letter}"
        activation_path = args.output_dir / f"{model_key}_manifolds.npz"
        if activation_path.exists():
            with np.load(activation_path) as saved:
                manifolds = {layer: saved[layer] for layer in LAYERS}
            model_checkpoint_path = checkpoint_path(n_classes, seed_letter)
        else:
            model, model_checkpoint_path = load_model(n_classes, seed_letter, device)
            manifolds = extract_manifolds(
                model,
                image_paths,
                batch_size=args.batch_size,
                device=device,
                projections=projections,
            )
            np.savez_compressed(activation_path, **manifolds)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        entry = results.setdefault(model_key, {
            "n_classes": n_classes,
            "seed": seed,
            "checkpoint": str(model_checkpoint_path),
        })
        for layer in LAYERS:
            layer_result = entry.setdefault(layer, {})
            if args.metrics in ("snr", "both") and "snr" not in layer_result:
                snr = manifold_snr(manifolds[layer], n_shots=5)
                layer_result["snr"] = {
                    "mean": snr["mean"],
                    "per_category": np.nanmean(snr["pairwise"], axis=1).tolist(),
                }
                save_json(results_path, results)
            if args.metrics in ("capacity", "both") and "capacity" not in layer_result:
                capacity = manifold_capacity(
                    manifolds[layer], n_probes=args.n_probes, seed=args.panel_seed + seed
                )
                layer_result["capacity"] = {
                    "mean": capacity["mean"],
                    "per_manifold": capacity["per_manifold"].tolist(),
                    "center_correlation": capacity["center_correlation"],
                    "center_rank": capacity["center_rank"],
                }
                save_json(results_path, results)

    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
