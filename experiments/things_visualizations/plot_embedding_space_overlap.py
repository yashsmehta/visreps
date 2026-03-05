"""
Overlapping UMAP embedding: images placed at raw UMAP coordinates.

Dense clusters overlap, peripheral images (most different) stand out clearly.
Render order: center-of-mass first, periphery last (outliers on top).

Input:  experiments/things_visualizations/data/things_viz_data.npz
Output: experiments/things_visualizations/figures/umap_embedding_overlap.png
"""

import os
import numpy as np
from PIL import Image

from experiments.things_visualizations.utils import (
    load_data, run_umap, load_thumbnail, coords_to_pixels,
    add_frame, add_title, stitch_panels, FIG_DIR,
)

CANVAS_SIZE = 3200
THUMB_SIZE = 80


def render_overlap_panel(coords, image_paths):
    """Render images at UMAP positions, center first, periphery on top."""
    canvas = Image.new("RGB", (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255))
    px, py = coords_to_pixels(coords, CANVAS_SIZE, THUMB_SIZE)

    # Sort: closest to centroid first -> farthest (most unique) last (on top)
    centroid = coords.mean(axis=0)
    order = np.argsort(np.sum((coords - centroid) ** 2, axis=1))

    half = THUMB_SIZE // 2
    for idx in order:
        thumb = load_thumbnail(str(image_paths[idx]), size=THUMB_SIZE,
                               border_width=1, border_color=(50, 50, 50))
        canvas.paste(thumb, (px[idx] - half, py[idx] - half))
    return canvas


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    data = load_data()
    rep_image_paths = data["rep_image_paths"]

    panels_config = [
        ("Behavioral Embedding", data["embeddings"]),
        ("CLIP 4-class Model", data["clip4_acts"]),
        ("1000-class Model", data["thousand_acts"]),
    ]

    rendered = []
    for title, feats in panels_config:
        print(f"Rendering {title}...")
        coords = run_umap(feats)
        panel = render_overlap_panel(coords, rep_image_paths)
        panel = add_title(add_frame(panel), title)
        rendered.append(panel)

        safe_name = title.lower().replace(" ", "_")
        panel_path = os.path.join(FIG_DIR, f"umap_overlap_{safe_name}.png")
        panel.save(panel_path, quality=95)
        print(f"  Saved: {panel_path}")

    final = stitch_panels(rendered)
    out = os.path.join(FIG_DIR, "umap_embedding_overlap.png")
    final.save(out, quality=95)
    print(f"\nSaved combined: {out} ({final.width}x{final.height})")


if __name__ == "__main__":
    main()
