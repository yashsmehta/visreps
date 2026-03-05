"""
Dense UMAP embedding mosaic: all ~1,854 THINGS concepts on a gapless grid.

Uses the Hungarian algorithm (linear assignment) to snap each image to the
nearest cell on a regular grid, creating a continuous mosaic.

Input:  experiments/things_visualizations/data/things_viz_data.npz
Output: experiments/things_visualizations/figures/umap_embedding_dense.png
"""

import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from PIL import Image

from experiments.things_visualizations.utils import (
    load_data, run_umap, load_thumbnail, add_frame, add_title, stitch_panels, FIG_DIR,
)

THUMB_SIZE = 64
BORDER_WIDTH = 1


def compute_grid_assignment(coords, n_rows, n_cols):
    """Assign images to grid cells using the Hungarian algorithm.

    Drops the most peripheral images (farthest from centroid) so that
    n_images == n_cells for a perfect 1:1 mapping.

    Returns: fill_map {(r,c): img_idx}
    """
    n_images = len(coords)
    n_cells = n_rows * n_cols
    n_drop = n_images - n_cells

    # Normalize coords to [0, 1]
    c = coords.copy()
    for d in range(2):
        mn, mx = c[:, d].min(), c[:, d].max()
        c[:, d] = (c[:, d] - mn) / (mx - mn) if mx > mn else 0.5

    # Drop peripheral images if needed
    keep_mask = np.ones(n_images, dtype=bool)
    if n_drop > 0:
        centroid = c.mean(axis=0)
        dist_to_center = np.sum((c - centroid) ** 2, axis=1)
        keep_mask[np.argsort(dist_to_center)[-n_drop:]] = False
        print(f"  Dropping {n_drop} most peripheral concepts")
        c = c[keep_mask]

    kept_indices = np.where(keep_mask)[0]

    # Grid cell centers
    grid_row = np.repeat(np.arange(n_rows), n_cols)
    grid_col = np.tile(np.arange(n_cols), n_rows)
    grid_centers = np.column_stack([
        grid_col / max(n_cols - 1, 1),
        grid_row / max(n_rows - 1, 1),
    ])

    print(f"  Running Hungarian algorithm ({len(c)} x {n_cells})...")
    cost = np.sum((c[:, None, :] - grid_centers[None, :, :]) ** 2, axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)

    fill_map = {}
    for kept_pos, cell_idx in zip(row_ind, col_ind):
        fill_map[(int(grid_row[cell_idx]), int(grid_col[cell_idx]))] = int(kept_indices[kept_pos])
    return fill_map


def render_grid_panel(fill_map, image_paths, n_rows, n_cols):
    """Render a fully-packed grid mosaic."""
    canvas = Image.new("RGB", (n_cols * THUMB_SIZE, n_rows * THUMB_SIZE), (255, 255, 255))
    for (r, c), img_idx in fill_map.items():
        thumb = load_thumbnail(str(image_paths[img_idx]), size=THUMB_SIZE,
                               border_width=BORDER_WIDTH, border_color=(210, 210, 210))
        canvas.paste(thumb, (c * THUMB_SIZE, r * THUMB_SIZE))
    return canvas


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    data = load_data()
    rep_image_paths = data["rep_image_paths"]
    n_concepts = data["embeddings"].shape[0]

    n_cols = n_rows = int(np.floor(np.sqrt(n_concepts)))
    print(f"Grid: {n_rows}x{n_cols} = {n_rows * n_cols} cells (dropping {n_concepts - n_rows * n_cols} outliers)")

    panels_config = [
        ("Behavioral Embedding", data["embeddings"]),
        ("CLIP 4-class Model", data["clip4_acts"]),
        ("1000-class Model", data["thousand_acts"]),
    ]

    rendered = []
    for title, feats in panels_config:
        print(f"\n{title}:")
        coords = run_umap(feats)
        fill_map = compute_grid_assignment(coords, n_rows, n_cols)
        panel = render_grid_panel(fill_map, rep_image_paths, n_rows, n_cols)
        panel = add_title(add_frame(panel), title)
        rendered.append(panel)

        safe_name = title.lower().replace(" ", "_")
        panel_path = os.path.join(FIG_DIR, f"umap_dense_{safe_name}.png")
        panel.save(panel_path, quality=95)
        print(f"  Saved: {panel_path}")

    final = stitch_panels(rendered)
    out = os.path.join(FIG_DIR, "umap_embedding_dense.png")
    final.save(out, quality=95)
    print(f"\nSaved combined: {out} ({final.width}x{final.height})")


if __name__ == "__main__":
    main()
