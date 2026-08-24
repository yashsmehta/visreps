"""Build a reusable manifest of held-out ImageNet images for manifold analyses.

The manifest references existing image files instead of copying 50,000 images.
Membership exactly follows ImageNetDataset's deterministic 80/20 split.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


def historical_holdout(
    imagenet_dir: Path, label_file: Path, *, train_ratio: float, split_seed: int
) -> list[tuple[str, int, str]]:
    """Reproduce the legacy ImageNetDataset global seed-42 80/20 split."""
    folder_labels = json.loads(label_file.read_text())
    samples = []
    for folder in os.listdir(imagenet_dir):
        folder_path = imagenet_dir / folder
        if (
            not folder.startswith("n")
            or not folder_path.is_dir()
            or folder not in folder_labels
        ):
            continue
        label = int(folder_labels[folder])
        for filename in os.listdir(folder_path):
            if filename.lower().endswith((".jpeg", ".jpg")):
                samples.append((str(folder_path / filename), label, filename))
    samples.sort(key=lambda sample: sample[2])
    generator = torch.Generator().manual_seed(split_seed)
    permutation = torch.randperm(len(samples), generator=generator).tolist()
    split_index = int(len(samples) * train_ratio)
    return [samples[index] for index in permutation[split_index:]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet-dir", type=Path, default=Path("/data/shared/datasets/imagenet"))
    parser.add_argument(
        "--label-file",
        type=Path,
        default=Path("datasets/obj_cls/imagenet/folder_labels.json"),
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--images-per-category", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/manifold_analysis/heldout_dataset/manifest.json"),
    )
    args = parser.parse_args()

    if not 0 < args.train_ratio < 1:
        parser.error("--train-ratio must be between zero and one")
    samples = historical_holdout(
        args.imagenet_dir,
        args.label_file,
        train_ratio=args.train_ratio,
        split_seed=args.split_seed,
    )
    by_category: dict[str, list[str]] = defaultdict(list)
    for path, _, _ in samples:
        by_category[Path(path).parent.name].append(path)

    rng = np.random.default_rng(args.seed)
    selected = {}
    for category in sorted(by_category):
        paths = by_category[category]
        if len(paths) < args.images_per_category:
            raise ValueError(
                f"{category} has {len(paths)} held-out images; "
                f"need {args.images_per_category}"
            )
        selected[category] = sorted(
            rng.choice(paths, args.images_per_category, replace=False).tolist()
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({
        "source": str(args.imagenet_dir),
        "split": (
            f"deterministic held-out {100 * (1 - args.train_ratio):g}% "
            f"(torch seed {args.split_seed})"
        ),
        "train_ratio": args.train_ratio,
        "split_seed": args.split_seed,
        "selection_seed": args.seed,
        "images_per_category": args.images_per_category,
        "n_categories": len(selected),
        "categories": selected,
    }, indent=2))
    print(f"Saved {len(selected)} categories and {sum(map(len, selected.values()))} paths to {args.output}")


if __name__ == "__main__":
    main()
