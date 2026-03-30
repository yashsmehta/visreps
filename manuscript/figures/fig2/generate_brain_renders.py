"""Generate pre-rendered brain ROI images for Figure 2 insets.

Uses nilearn to render human cortical surfaces (fsaverage, inflated, left
hemisphere lateral view) with Destrieux atlas ROIs highlighted.  Outputs
transparent PNGs into the assets/ directory.

Macaque brain insets use the Scidraw SVG with ellipse overlays at runtime
(handled by schematic_utils.py), so no macaque renders are generated here.

Usage:
    python manuscript/figures/fig2/generate_brain_renders.py
"""

import io
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from nilearn import datasets
from nilearn.plotting import plot_surf_roi
from nilearn.surface import load_surf_data
from PIL import Image

ASSETS_DIR = Path("manuscript/figures/fig2/assets")

# Destrieux surface-label indices (left hemisphere) for each ROI
ROI_LABELS = {
    "early": {
        "label_ids": [2, 11, 20, 43, 45],
        # G_and_S_occipital_inf, G_cuneus, G_occipital_sup,
        # Pole_occipital, S_calcarine
        "output": "brain_early_visual.png",
    },
    "ventral": {
        "label_ids": [21, 22, 37, 51, 52, 61, 62],
        # G_oc-temp_lat-fusifor, G_oc-temp_med-Lingual, G_temporal_inf,
        # S_collat_transv_ant, S_collat_transv_post,
        # S_oc-temp_lat, S_oc-temp_med_and_Lingual
        "output": "brain_ventral_visual.png",
    },
}


def render_roi(fsaverage, labels_left, label_ids, output_path):
    """Render a single ROI on inflated left hemisphere and save as PNG."""
    roi_map = np.zeros(len(labels_left), dtype=float)
    for lbl in label_ids:
        roi_map[labels_left == lbl] = 1.0

    fig, ax = plt.subplots(1, 1, subplot_kw={"projection": "3d"},
                           figsize=(3.5, 3))
    plot_surf_roi(
        fsaverage["infl_left"], roi_map,
        hemi="left", view="lateral",
        bg_map=fsaverage["sulc_left"],
        bg_on_data=True,
        axes=ax,
        colorbar=False,
        cmap="Reds",
        alpha=0.85,
    )
    ax.set_axis_off()
    fig.patch.set_alpha(0.0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight",
                transparent=True, pad_inches=0.01)
    plt.close(fig)

    buf.seek(0)
    Image.open(buf).convert("RGBA").save(output_path)
    print(f"Saved {output_path}  ({np.sum(roi_map > 0)} vertices)")


def main():
    warnings.filterwarnings("ignore", module="nilearn")
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    fsaverage = datasets.fetch_surf_fsaverage()
    destrieux = datasets.fetch_atlas_surf_destrieux()
    labels_left = load_surf_data(destrieux["map_left"]).astype(int)

    for name, spec in ROI_LABELS.items():
        render_roi(fsaverage, labels_left, spec["label_ids"],
                   ASSETS_DIR / spec["output"])


if __name__ == "__main__":
    main()
