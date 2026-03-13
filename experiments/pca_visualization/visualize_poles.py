"""Visualize the most and least activating ImageNet images for principal components.

Usage (single PC):
    python experiments/pca_visualization/visualize_poles.py --model alexnet --pc 1

Usage (all PCs — publication figure):
    python experiments/pca_visualization/visualize_poles.py --model alexnet --all_pcs
"""

import os
import sys
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

POLES_DIR = os.path.join(PROJECT_ROOT, "datasets", "obj_cls", "imagenet", "pca_poles")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "experiments", "pca_visualization", "figures")

# Map model names to their poles CSV filenames
POLES_FILES = {
    "alexnet": "pca_poles_alexnet.csv",
    "clip": "pca_poles_clip_vit.csv",
    "dino": "pca_poles_dino.csv",
    "vit": "pca_poles_vit.csv",
}

MODEL_DISPLAY_NAMES = {
    "alexnet": "AlexNet",
    "clip": "CLIP ViT-L/14",
    "dino": "DINOv3 ViT-L",
    "vit": "ViT-L (supervised)",
}


def load_poles_data(model: str) -> pd.DataFrame:
    """Load the precomputed poles CSV for a given model."""
    filename = POLES_FILES[model]
    path = os.path.join(POLES_DIR, filename)
    if not os.path.exists(path):
        available = [m for m, f in POLES_FILES.items()
                     if os.path.exists(os.path.join(POLES_DIR, f))]
        raise FileNotFoundError(
            f"Poles file not found: {path}\n"
            f"Available models with poles data: {available}\n"
            f"Generate missing poles with: python experiments/pca_visualization/generate_poles.py"
        )
    return pd.read_csv(path)


def load_image(image_file: str, imagenet_dir: str, thumb_size: int = 224) -> Image.Image:
    """Load an ImageNet image, center-crop to square, and resize to thumbnail."""
    class_id = image_file.split("_")[0]
    path = os.path.join(imagenet_dir, class_id, image_file)
    img = Image.open(path).convert("RGB")
    # Center-crop to square
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img.resize((thumb_size, thumb_size), Image.LANCZOS)


def make_figure(
    high_images: list[Image.Image],
    low_images: list[Image.Image],
    model: str,
    pc: int,
    n_images: int,
) -> plt.Figure:
    """Create a figure with two labeled blocks (most/least activating), each a 2-row grid."""
    n_cols = min(n_images, 10)
    n_img_rows = (n_images + n_cols - 1) // n_cols  # rows per block (2 for 20 images)

    cell = 0.85         # inches per image cell
    pad = 0.02          # gap between cells (fraction of cell)
    label_space = 0.35  # inches for each block label
    block_gap = 0.45    # inches between the two blocks
    title_space = 0.45  # inches for the figure title

    fig_w = n_cols * cell
    fig_h = (2 * n_img_rows * cell) + (2 * label_space) + block_gap + title_space

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")
    display_name = MODEL_DISPLAY_NAMES.get(model, model)

    # Title
    fig.text(
        0.5, 1.0 - 0.3 * title_space / fig_h,
        f"PC {pc} — {display_name}",
        ha="center", va="top",
        fontsize=14, fontweight="bold", fontfamily="sans-serif",
    )

    # Compute vertical positions for each block (in figure coords)
    top_block_top = 1.0 - title_space / fig_h
    bot_block_top = top_block_top - (n_img_rows * cell + label_space) / fig_h - block_gap / fig_h

    for block_idx, (images, label) in enumerate([
        (high_images, "Most Activating"),
        (low_images, "Least Activating"),
    ]):
        block_top = top_block_top if block_idx == 0 else bot_block_top
        row_h = cell / fig_h
        col_w = 1.0 / n_cols

        # Block label
        fig.text(
            0.5, block_top - 0.005,
            label,
            ha="center", va="top",
            fontsize=11, fontweight="semibold", fontfamily="sans-serif",
            color="#444444",
        )

        grid_top = block_top - label_space / fig_h

        for img_idx, img in enumerate(images):
            r = img_idx // n_cols
            c = img_idx % n_cols
            left = c * col_w + pad * col_w
            bottom = grid_top - (r + 1) * row_h + pad * row_h
            w = col_w * (1 - 2 * pad)
            h = row_h * (1 - 2 * pad)

            ax = fig.add_axes([left, bottom, w, h])
            ax.imshow(img)
            ax.set_axis_off()

    return fig


def load_variance_explained(model: str) -> list[float]:
    """Load per-PC variance explained (%) from eigenvectors file."""
    eigvec_path = os.path.join(
        PROJECT_ROOT, "datasets", "obj_cls", "imagenet", f"eigenvectors_{model}.npz"
    )
    data = np.load(eigvec_path)
    return (data["eigenvalues"] / data["total_variance"] * 100).tolist()


def make_allpcs_figure(
    all_high: dict[int, list[Image.Image]],
    all_low: dict[int, list[Image.Image]],
    model: str,
    var_explained: list[float],
    n_per_pole: int = 5,
) -> plt.Figure:
    """Publication-quality table layout: 6 rows (PC1-PC6), N most + N least per row."""
    import matplotlib.gridspec as gridspec

    sns.set_theme(style="white", context="paper", font_scale=1.2)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    })

    n_pcs = len(all_high)
    n_cols = n_per_pole
    # Grid: n_pcs rows × (n_cols + 1 gap + n_cols) columns
    # The gap column has width_ratio ~0.3 to create visual separation
    total_grid_cols = 2 * n_cols + 1  # 7 high + 1 gap + 7 low = 15
    width_ratios = [1.0] * n_cols + [0.3] + [1.0] * n_cols

    cell_size = 0.9  # inches per image cell
    fig_w = sum(width_ratios) * cell_size + 1.2  # +1.2 for left label margin
    fig_h = n_pcs * cell_size + 0.7               # +0.7 for header

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    gs = gridspec.GridSpec(
        n_pcs, total_grid_cols,
        figure=fig,
        left=1.2 / fig_w,       # left margin for labels
        right=0.995,
        top=1.0 - 0.55 / fig_h,  # top margin for headers
        bottom=0.02,
        wspace=0.05,
        hspace=0.15,
        width_ratios=width_ratios,
    )

    display_name = MODEL_DISPLAY_NAMES.get(model, model)
    pcs = sorted(all_high.keys())

    # Column headers
    # "Most Activating" centered over left block, "Least Activating" over right
    gs_left = 1.2 / fig_w
    gs_right = 0.995
    gs_content_w = gs_right - gs_left
    total_ratio = sum(width_ratios)
    left_block_w = sum(width_ratios[:n_cols]) / total_ratio * gs_content_w
    gap_w = width_ratios[n_cols] / total_ratio * gs_content_w
    right_block_w = sum(width_ratios[n_cols + 1:]) / total_ratio * gs_content_w

    header_y = 1.0 - 0.15 / fig_h
    fig.text(
        gs_left + left_block_w / 2, header_y,
        "Most Activating",
        ha="center", va="bottom",
        fontsize=15, fontweight="bold", fontfamily="sans-serif", color="#1a1a1a",
    )
    fig.text(
        gs_left + left_block_w + gap_w + right_block_w / 2, header_y,
        "Least Activating",
        ha="center", va="bottom",
        fontsize=15, fontweight="bold", fontfamily="sans-serif", color="#1a1a1a",
    )

    for row_idx, pc in enumerate(pcs):
        var_pct = var_explained[pc - 1]

        # Get vertical center of this row for label placement
        # Use the first cell's position as reference
        tmp_ax = fig.add_subplot(gs[row_idx, 0])
        bbox = tmp_ax.get_position()
        row_center_y = (bbox.y0 + bbox.y1) / 2
        tmp_ax.remove()

        # PC label
        fig.text(
            gs_left - 0.01, row_center_y + 0.01,
            f"PC {pc}",
            ha="right", va="center",
            fontsize=13, fontweight="bold", fontfamily="sans-serif", color="#2a2a2a",
        )
        fig.text(
            gs_left - 0.01, row_center_y - 0.025,
            f"{var_pct:.1f}% var.",
            ha="right", va="center",
            fontsize=9.5, fontfamily="sans-serif", color="#888888",
        )

        # High-pole images (columns 0 to n_cols-1)
        for col, img in enumerate(all_high[pc]):
            ax = fig.add_subplot(gs[row_idx, col])
            ax.imshow(img, interpolation="lanczos")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#999999")
                spine.set_linewidth(0.8)
            ax.set_xticks([])
            ax.set_yticks([])

        # Gap column (n_cols) — leave empty

        # Low-pole images (columns n_cols+1 to end)
        for col, img in enumerate(all_low[pc]):
            ax = fig.add_subplot(gs[row_idx, n_cols + 1 + col])
            ax.imshow(img, interpolation="lanczos")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#999999")
                spine.set_linewidth(0.8)
            ax.set_xticks([])
            ax.set_yticks([])

    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize PC pole images")
    parser.add_argument("--model", type=str, required=True, choices=list(POLES_FILES.keys()))
    parser.add_argument("--pc", type=int, default=None, help="PC index (1-based)")
    parser.add_argument("--all_pcs", action="store_true", help="Publication figure: all PCs 1-6")
    parser.add_argument("--n_images", type=int, default=20,
                        help="Images per pole (single-PC mode, default 20)")
    parser.add_argument("--n_per_pole", type=int, default=5,
                        help="Images per pole (all-PCs mode, default 5)")
    args = parser.parse_args()

    if not args.all_pcs and args.pc is None:
        parser.error("Provide either --pc <N> or --all_pcs")

    imagenet_dir = os.environ.get("IMAGENET_DATA_DIR")
    if not imagenet_dir:
        raise EnvironmentError("IMAGENET_DATA_DIR not set. Source .env first.")

    df = load_poles_data(args.model)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    if args.all_pcs:
        # --- Publication figure: all 6 PCs ---
        var_explained = load_variance_explained(args.model)
        all_high, all_low = {}, {}
        for pc in range(1, 7):
            df_pc = df[df["pc"] == pc]
            high_df = df_pc[df_pc["pole"] == "high"].head(args.n_per_pole)
            low_df = df_pc[df_pc["pole"] == "low"].head(args.n_per_pole)
            print(f"  PC {pc}: loading {len(high_df)}+{len(low_df)} images...")
            all_high[pc] = [load_image(r["image_file"], imagenet_dir)
                            for _, r in high_df.iterrows()]
            all_low[pc] = [load_image(r["image_file"], imagenet_dir)
                           for _, r in low_df.iterrows()]

        fig = make_allpcs_figure(all_high, all_low, args.model, var_explained, args.n_per_pole)
        out_path = os.path.join(FIGURES_DIR, f"all_pcs_{args.model}.png")
        fig.savefig(out_path, dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none")
        plt.close(fig)
        print(f"Saved: {out_path}")

    else:
        # --- Single-PC figure ---
        df_pc = df[df["pc"] == args.pc]
        if df_pc.empty:
            available_pcs = sorted(df["pc"].unique())
            raise ValueError(f"PC {args.pc} not found. Available PCs: {available_pcs}")

        high_df = df_pc[df_pc["pole"] == "high"].head(args.n_images)
        low_df = df_pc[df_pc["pole"] == "low"].head(args.n_images)

        print(f"Loading {len(high_df)} most-activating and {len(low_df)} least-activating images "
              f"for PC {args.pc} ({args.model})...")

        high_images = [load_image(row["image_file"], imagenet_dir)
                       for _, row in high_df.iterrows()]
        low_images = [load_image(row["image_file"], imagenet_dir)
                      for _, row in low_df.iterrows()]

        fig = make_figure(high_images, low_images, args.model, args.pc, args.n_images)
        out_path = os.path.join(FIGURES_DIR, f"pc{args.pc}_{args.model}.png")
        fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
