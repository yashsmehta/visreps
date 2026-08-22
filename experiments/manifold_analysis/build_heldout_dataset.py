"""Build a reusable manifest of held-out ImageNet images for manifold analyses.

The manifest references existing image files instead of copying 50,000 images.
Membership exactly follows ImageNetDataset's deterministic 80/20 split.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from visreps.dataloaders.obj_cls import ImageNetDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet-dir", type=Path, default=Path("/data/shared/datasets/imagenet"))
    parser.add_argument("--images-per-category", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/manifold_analysis/heldout_dataset/manifest.json"),
    )
    args = parser.parse_args()

    dataset = ImageNetDataset(str(args.imagenet_dir), split="test", transform=None)
    by_category: dict[str, list[str]] = defaultdict(list)
    for path, _, _ in dataset.samples:
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
        "split": "deterministic held-out 20% (torch seed 42)",
        "selection_seed": args.seed,
        "images_per_category": args.images_per_category,
        "n_categories": len(selected),
        "categories": selected,
    }, indent=2))
    print(f"Saved {len(selected)} categories and {sum(map(len, selected.values()))} paths to {args.output}")


if __name__ == "__main__":
    main()
