"""Supplementary Figure S5: PC Axis Interpretation.

Shows the most-activating and least-activating ImageNet images for the top
principal components of each PCA source model (AlexNet and CLIP).

Layout: Two side-by-side panels (AlexNet | CLIP), each showing 6 PCs with
5 most-activating and 5 least-activating images per PC.

Usage:
    python manuscript/figures/extended_data/supp_s5_pc_poles.py
"""

import os
import sys

sys.path.insert(0, ".")

from dotenv import load_dotenv
load_dotenv(".env")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

from manuscript.figures.fig_utils import setup_style

# Config
OUTPUT = "manuscript/figures/extended_data/S5_pc_poles/S5_pc_poles.png"
POLES_DIR = "datasets/obj_cls/imagenet/pca_poles"
EIGVEC_DIR = "datasets/obj_cls/imagenet"

MODELS = {
    "alexnet": {
        "poles_file": "pca_poles_alexnet.csv",
        "eigvec_file": "eigenvectors_alexnet.npz",
        "display": "AlexNet representations",
    },
    "clip": {
        "poles_file": "pca_poles_clip_vit.csv",
        "eigvec_file": "eigenvectors_clip.npz",
        "display": "CLIP representations",
    },
}

# Sub-header colors
LEAST_ACTIVATING_COLOR = "#1b5e20"   # dark green
MOST_ACTIVATING_COLOR = "#e65100"    # deep orange

N_PCS = 6
N_PER_POLE = 5
THUMB_SIZE = 224


def load_image(image_file: str, imagenet_dir: str) -> Image.Image:
    """Load an ImageNet image, center-crop to square, and resize."""
    class_id = image_file.split("_")[0]
    path = os.path.join(imagenet_dir, class_id, image_file)
    img = Image.open(path).convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img.resize((THUMB_SIZE, THUMB_SIZE), Image.LANCZOS)


def load_model_data(model_key: str, imagenet_dir: str):
    """Load pole images and variance explained for a model."""
    cfg = MODELS[model_key]
    poles_path = os.path.join(POLES_DIR, cfg["poles_file"])
    eigvec_path = os.path.join(EIGVEC_DIR, cfg["eigvec_file"])

    df = pd.read_csv(poles_path)
    data = np.load(eigvec_path)
    var_explained = (data["eigenvalues"] / data["total_variance"] * 100).tolist()

    all_high, all_low = {}, {}
    for pc in range(1, N_PCS + 1):
        df_pc = df[df["pc"] == pc]
        high_df = df_pc[df_pc["pole"] == "high"].head(N_PER_POLE)
        low_df = df_pc[df_pc["pole"] == "low"].head(N_PER_POLE)
        print(f"  {cfg['display']} PC {pc}: loading {len(high_df)}+{len(low_df)} images...")
        all_high[pc] = [load_image(r["image_file"], imagenet_dir)
                        for _, r in high_df.iterrows()]
        all_low[pc] = [load_image(r["image_file"], imagenet_dir)
                       for _, r in low_df.iterrows()]

    return all_high, all_low, var_explained


def make_figure(model_data: dict) -> plt.Figure:
    """Create composite figure with both models side by side."""
    # Layout: 5 low + gap + 5 high per block, with PC labels before each block
    n_cols_per_block = 2 * N_PER_POLE + 1  # 5 low + 1 gap + 5 high = 11

    cell_size = 0.75
    gap_ratio = 0.3
    model_gap = 1.0
    pc_label_ratio = 0.6  # space for PC label column per model
    left_margin = 0.1
    top_margin = 0.7
    bottom_margin = 0.1

    block_ratios = [1.0] * N_PER_POLE + [gap_ratio] + [1.0] * N_PER_POLE
    model_gap_ratio = model_gap / cell_size
    # Each model block: pc_label + 11 image columns
    all_ratios = ([pc_label_ratio] + block_ratios
                  + [model_gap_ratio]
                  + [pc_label_ratio] + block_ratios)

    fig_w = left_margin + sum(r * cell_size for r in all_ratios)
    fig_h = top_margin + N_PCS * cell_size + bottom_margin

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    gs = gridspec.GridSpec(
        N_PCS, len(all_ratios),
        figure=fig,
        left=left_margin / fig_w,
        right=0.995,
        top=1.0 - top_margin / fig_h,
        bottom=bottom_margin / fig_h,
        wspace=0.04,
        hspace=0.12,
        width_ratios=all_ratios,
    )

    model_keys = list(model_data.keys())
    # Column offsets: model 0 starts at col 0 (label) + 1 (images)
    # model 1 starts after model 0's block + gap
    block_width = 1 + n_cols_per_block  # pc_label col + 11 image cols
    model_offsets = [0, block_width + 1]  # +1 for the model gap column

    for model_idx, model_key in enumerate(model_keys):
        all_high, all_low, var_explained = model_data[model_key]
        display = MODELS[model_key]["display"]

        base_col = model_offsets[model_idx]
        pc_label_col = base_col          # column for PC labels
        img_offset = base_col + 1        # first image column

        # Model title
        block_left = gs[0, img_offset].get_position(fig).x0
        block_right = gs[0, img_offset + n_cols_per_block - 1].get_position(fig).x1
        title_x = (block_left + block_right) / 2
        title_y = 1.0 - 0.15 / fig_h

        fig.text(title_x, title_y, display,
                 ha="center", va="bottom",
                 fontsize=12, fontweight="bold")

        # Sub-headers: Least Activating (left, green) | Most Activating (right, orange)
        low_left = gs[0, img_offset].get_position(fig).x0
        low_right = gs[0, img_offset + N_PER_POLE - 1].get_position(fig).x1
        high_left = gs[0, img_offset + N_PER_POLE + 1].get_position(fig).x0
        high_right = gs[0, img_offset + n_cols_per_block - 1].get_position(fig).x1

        header_y = 1.0 - 0.40 / fig_h
        fig.text((low_left + low_right) / 2, header_y, "Least Activating",
                 ha="center", va="bottom", fontsize=8,
                 fontweight="bold", color=LEAST_ACTIVATING_COLOR)
        fig.text((high_left + high_right) / 2, header_y, "Most Activating",
                 ha="center", va="bottom", fontsize=8,
                 fontweight="bold", color=MOST_ACTIVATING_COLOR)

        for row_idx, pc in enumerate(range(1, N_PCS + 1)):
            # PC label + variance % for EVERY model
            tmp_ax = fig.add_subplot(gs[row_idx, pc_label_col])
            bbox = tmp_ax.get_position()
            row_center_y = (bbox.y0 + bbox.y1) / 2
            label_x = (bbox.x0 + bbox.x1) / 2
            tmp_ax.remove()

            var_pct = var_explained[pc - 1]
            fig.text(label_x, row_center_y + 0.008,
                     f"PC {pc}",
                     ha="center", va="center",
                     fontsize=10, fontweight="bold", color="#2a2a2a")
            fig.text(label_x, row_center_y - 0.018,
                     f"{var_pct:.1f}%",
                     ha="center", va="center",
                     fontsize=7.5, color="#888888")

            # Low-pole images (LEFT side)
            for col, img in enumerate(all_low[pc]):
                ax = fig.add_subplot(gs[row_idx, img_offset + col])
                ax.imshow(img, interpolation="lanczos")
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color("#cccccc")
                    spine.set_linewidth(0.5)
                ax.set_xticks([])
                ax.set_yticks([])

            # High-pole images (RIGHT side)
            for col, img in enumerate(all_high[pc]):
                ax = fig.add_subplot(gs[row_idx, img_offset + N_PER_POLE + 1 + col])
                ax.imshow(img, interpolation="lanczos")
                for spine in ax.spines.values():
                    spine.set_visible(True)
                    spine.set_color("#cccccc")
                    spine.set_linewidth(0.5)
                ax.set_xticks([])
                ax.set_yticks([])

    return fig


def main():
    setup_style()

    imagenet_dir = os.environ.get("IMAGENET_DATA_DIR")
    if not imagenet_dir:
        raise EnvironmentError("IMAGENET_DATA_DIR not set. Source .env first.")

    print("Loading pole images for each model...")
    model_data = {}
    for model_key in MODELS:
        poles_path = os.path.join(POLES_DIR, MODELS[model_key]["poles_file"])
        if not os.path.exists(poles_path):
            print(f"  WARNING: Poles file not found for {model_key}: {poles_path}")
            continue
        all_high, all_low, var_explained = load_model_data(model_key, imagenet_dir)
        model_data[model_key] = (all_high, all_low, var_explained)

    if not model_data:
        raise FileNotFoundError("No pole data found for any model.")

    print(f"\nCreating figure with {len(model_data)} model(s)...")
    fig = make_figure(model_data)

    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
