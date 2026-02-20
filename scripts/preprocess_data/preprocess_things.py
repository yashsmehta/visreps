"""Preprocess THINGS behavioral dataset: collect images and concept embeddings.

Scans the THINGS image directory, groups images by concept, and pairs them with
the 66-dimensional behavioral embeddings from Hebart et al. (2023).

Saves to datasets/neural/things/things_split.pkl with structure:
    {
        "embeddings":  {concept: np.ndarray(66,)},
        "image_ids":   {concept: [stimulus_id, ...]},
        "image_paths": {stimulus_id: absolute_path},
    }

The 80/20 concept-level train/test split for layer selection vs evaluation
is done at runtime in evals.py (not here).

Usage:
    python scripts/preprocess_data/preprocess_things.py
"""

import os
import pickle
from pathlib import Path

import numpy as np
from bonner.datasets.hebart2023_things_data.behavior import load_embeddings

SAVE_PATH = "datasets/neural/things/things_split.pkl"


def main():
    # Load concept-level embeddings (1854 × 66)
    beh_xr = load_embeddings()
    embeddings = {
        str(obj): beh_xr.sel(object=obj).values.astype(np.float32)
        for obj in beh_xr["object"].values
    }
    print(f"Loaded {len(embeddings)} concept embeddings")

    # Group images by concept from filesystem
    things_root = Path(
        os.environ.get(
            "BONNER_DATASETS_HOME", Path.home() / ".cache" / "bonner-datasets"
        ),
        "hebart2019.things",
        "images",
        "object_images",
    )
    image_paths = {}
    image_ids = {}

    for concept_dir in sorted(things_root.iterdir()):
        concept = concept_dir.name
        if not concept_dir.is_dir() or concept not in embeddings:
            continue
        imgs = [f.stem for f in sorted(concept_dir.glob("*.jpg"))]
        if imgs:
            image_ids[concept] = imgs
            for stem in imgs:
                image_paths[stem] = str(concept_dir / f"{stem}.jpg")

    n_images = sum(len(v) for v in image_ids.values())
    print(f"Found {n_images} images across {len(image_ids)} concepts")

    # Save
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    with open(SAVE_PATH, "wb") as f:
        pickle.dump(
            {
                "embeddings": embeddings,
                "image_ids": image_ids,
                "image_paths": image_paths,
            },
            f,
        )
    print(f"Saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()
