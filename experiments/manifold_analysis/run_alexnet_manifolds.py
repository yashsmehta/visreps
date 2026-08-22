"""Compute FC2 manifold geometry for ImageNet-pretrained AlexNet."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv
from PIL import Image
from torchvision import models
from torchvision.datasets import ImageFolder

from experiments.manifold_analysis.manifold_capacity import manifold_capacity
from experiments.manifold_analysis.manifold_snr import manifold_snr


def sample_images(
    imagenet_dir: Path,
    n_categories: int,
    images_per_category: int,
    seed: int,
) -> tuple[list[str], list[list[str]]]:
    """Select a reproducible, balanced ImageNet panel."""
    dataset = ImageFolder(imagenet_dir)
    by_category: list[list[str]] = [[] for _ in dataset.classes]
    for path, category in dataset.samples:
        by_category[category].append(path)

    eligible = [i for i, paths in enumerate(by_category) if len(paths) >= images_per_category]
    if len(eligible) < n_categories:
        raise ValueError(
            f"only {len(eligible)} categories contain {images_per_category} images"
        )

    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(eligible, n_categories, replace=False))
    paths = [
        sorted(rng.choice(by_category[i], images_per_category, replace=False).tolist())
        for i in selected
    ]
    return [dataset.classes[i] for i in selected], paths


@torch.inference_mode()
def extract_fc2(
    image_paths: list[list[str]],
    *,
    batch_size: int,
    device: torch.device,
    projection_seed: int,
) -> np.ndarray:
    """Extract post-ReLU FC2 activations and project 4,096 -> 2,048."""
    weights = models.AlexNet_Weights.IMAGENET1K_V1
    model = models.alexnet(weights=weights).eval().to(device)
    preprocess = weights.transforms()

    activations = []
    for category_paths in image_paths:
        category_batches = []
        for start in range(0, len(category_paths), batch_size):
            images = [
                preprocess(Image.open(path).convert("RGB"))
                for path in category_paths[start : start + batch_size]
            ]
            x = torch.stack(images).to(device)
            x = model.features(x)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            x = model.classifier[:6](x)  # dropout, FC1, ReLU, dropout, FC2, ReLU
            category_batches.append(x.cpu())
        activations.append(torch.cat(category_batches))

    raw = torch.stack(activations)  # categories, images, 4096
    generator = torch.Generator().manual_seed(projection_seed)
    projection = torch.randn(4096, 2048, generator=generator) / np.sqrt(2048)
    projected = raw @ projection
    return projected.permute(0, 2, 1).numpy()


def json_ready(result: dict[str, object]) -> dict[str, object]:
    return {
        key: value.tolist() if isinstance(value, np.ndarray) else value
        for key, value in result.items()
    }


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--imagenet-dir",
        type=Path,
        default=os.environ.get("IMAGENET_DATA_DIR"),
        help="ImageNet ImageFolder root (defaults to IMAGENET_DATA_DIR)",
    )
    parser.add_argument("--metric", choices=("snr", "capacity", "both"), default="both")
    parser.add_argument("--n-categories", type=int, default=50)
    parser.add_argument("--images-per-category", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-shots", type=int, default=5)
    parser.add_argument("--n-probes", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("alexnet_manifolds.json"))
    args = parser.parse_args()

    if args.imagenet_dir is None:
        parser.error("provide --imagenet-dir or set IMAGENET_DATA_DIR")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    categories, paths = sample_images(
        args.imagenet_dir, args.n_categories, args.images_per_category, args.seed
    )
    manifolds = extract_fc2(
        paths,
        batch_size=args.batch_size,
        device=device,
        projection_seed=args.seed,
    )

    results: dict[str, object] = {
        "model": "torchvision AlexNet IMAGENET1K_V1",
        "layer": "FC2 post-ReLU",
        "categories": categories,
        "image_paths": paths,
        "manifold_shape": list(manifolds.shape),
        "seed": args.seed,
    }
    if args.metric in ("snr", "both"):
        results["snr"] = json_ready(manifold_snr(manifolds, args.n_shots))
    if args.metric in ("capacity", "both"):
        results["capacity"] = json_ready(
            manifold_capacity(manifolds, n_probes=args.n_probes, seed=args.seed)
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2))
    print(json.dumps({key: value["mean"] for key, value in results.items() if key in ("snr", "capacity")}, indent=2))
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
