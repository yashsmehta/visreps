"""
UMAP embedding space with image thumbnails: Behavioral | CLIP 4-class | 1000-class.

Grid-subsamples concepts for even spatial coverage and places one representative
image per concept at its UMAP coordinates.

Input:  experiments/things_visualizations/data/things_viz_data.npz
Output: experiments/things_visualizations/figures/umap_embedding.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from experiments.things_visualizations.utils import (
    load_data, run_umap, grid_subsample, load_thumbnail, setup_umap_axes, FIG_DIR,
)

GRID_SIZE = 5
THUMB_SIZE = 64
THUMB_ZOOM = 0.5


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    data = load_data()

    panels = [
        ("Behavioral", data["embeddings"]),
        ("CLIP 4-class", data["clip4_acts"]),
        ("1000-class", data["thousand_acts"]),
    ]

    print("Running UMAP...")
    umap_coords = {title: run_umap(feats) for title, feats in panels}

    display_idx = grid_subsample(umap_coords["Behavioral"], GRID_SIZE)
    print(f"Grid-sampled {len(display_idx)} concepts")

    fig, axes = plt.subplots(1, 3, figsize=(21, 7.5), gridspec_kw={"wspace": 0.03})
    for ax, (title, _) in zip(axes, panels):
        coords = umap_coords[title]
        setup_umap_axes(ax, coords, title)
        for idx in display_idx:
            thumb = load_thumbnail(str(data["rep_image_paths"][idx]), size=THUMB_SIZE)
            im = OffsetImage(np.array(thumb), zoom=THUMB_ZOOM)
            ax.add_artist(AnnotationBbox(im, (coords[idx, 0], coords[idx, 1]),
                                         frameon=False, pad=0.0, zorder=2))

    out = os.path.join(FIG_DIR, "umap_embedding.png")
    fig.savefig(out, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
