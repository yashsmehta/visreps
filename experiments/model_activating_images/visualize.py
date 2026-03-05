"""Visualize the most-activating ImageNet images per output class.

Reads the rankings CSV produced by generate_rankings.py and creates a
publication-quality figure: one row per class, N images per row.

Usage:
    python experiments/model_activating_images/visualize.py
    python experiments/model_activating_images/visualize.py --csv_path rankings_4class.csv --labels "Artifacts" "Natural" "Scenes" "Animals"
"""

import os
import sys
import argparse

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIGURES_DIR = os.path.join(SCRIPT_DIR, "figures")

# Colorblind-friendly palette (up to 8 classes)
CLASS_COLORS = ["#3B7DD8", "#E05252", "#43A047", "#F4A236",
                "#8E44AD", "#16A085", "#D35400", "#2C3E50"]


def load_image(image_file: str, imagenet_dir: str, thumb_size: int = 224) -> Image.Image:
    """Load an ImageNet image, center-crop to square, and resize."""
    class_id = image_file.split("_")[0]
    path = os.path.join(imagenet_dir, class_id, image_file)
    img = Image.open(path).convert("RGB")
    w, h = img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    img = img.crop((left, top, left + side, top + side))
    return img.resize((thumb_size, thumb_size), Image.LANCZOS)


def make_figure(images_per_class, class_labels, n_images):
    """Publication figure: one row per class with colored accent bars."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    })

    n_classes = len(images_per_class)
    colors = CLASS_COLORS[:n_classes]

    cell_size = 1.15
    row_gap = 0.35
    left_margin = 1.4
    top_margin = 0.7
    bottom_margin = 0.25
    accent_bar_w = 0.06

    fig_w = n_images * cell_size + left_margin + 0.2
    fig_h = n_classes * (cell_size + row_gap) - row_gap + top_margin + bottom_margin

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor="white")

    gs = gridspec.GridSpec(
        n_classes, n_images, figure=fig,
        left=left_margin / fig_w, right=0.995,
        top=1.0 - top_margin / fig_h, bottom=bottom_margin / fig_h,
        wspace=0.04, hspace=row_gap / (cell_size + row_gap) * 1.1,
    )

    gs_left = left_margin / fig_w

    # Header
    title_x = (gs_left + 0.995) / 2
    title_y = 1.0 - 0.22 / fig_h
    fig.text(title_x, title_y, "Most Activating Images",
             ha="center", va="bottom", fontsize=17, fontweight="bold", color="#1a1a1a")

    # Divider
    divider_y = 1.0 - top_margin / fig_h + 0.015
    fig.lines.append(Line2D(
        [gs_left - accent_bar_w - 0.01, 0.995], [divider_y, divider_y],
        transform=fig.transFigure, color="#dddddd", linewidth=0.7, clip_on=False,
    ))

    for row_idx in range(n_classes):
        images = images_per_class[row_idx]
        color = colors[row_idx]
        label = class_labels.get(row_idx, f"Class {row_idx}")

        # Probe row position
        tmp_ax = fig.add_subplot(gs[row_idx, 0])
        bbox = tmp_ax.get_position()
        tmp_ax.remove()
        row_center_y = (bbox.y0 + bbox.y1) / 2

        # Row background tint
        tmp_ax2 = fig.add_subplot(gs[row_idx, -1])
        bbox_last = tmp_ax2.get_position()
        tmp_ax2.remove()

        fig.patches.append(FancyBboxPatch(
            (gs_left - 0.005, bbox.y0 - 0.008),
            (bbox_last.x1 - gs_left) + 0.01, (bbox.y1 - bbox.y0) + 0.016,
            boxstyle="round,pad=0.005", facecolor=color, edgecolor="none",
            alpha=0.04, transform=fig.transFigure, clip_on=False, zorder=0,
        ))

        # Accent bar
        bar_x = gs_left - accent_bar_w - 0.006
        fig.patches.append(FancyBboxPatch(
            (bar_x, bbox.y0 - 0.008), accent_bar_w * 0.35,
            (bbox.y1 - bbox.y0) + 0.016,
            boxstyle="round,pad=0.002", facecolor=color, edgecolor="none",
            alpha=0.9, transform=fig.transFigure, clip_on=False,
        ))

        # Class label
        fig.text(bar_x - 0.01, row_center_y, label,
                 ha="right", va="center", fontsize=13, fontweight="bold", color=color)

        # Images
        for col, img in enumerate(images[:n_images]):
            ax = fig.add_subplot(gs[row_idx, col])
            ax.imshow(img, interpolation="lanczos")
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#d0d0d0")
                spine.set_linewidth(0.6)
            ax.spines["top"].set_color(color)
            ax.spines["top"].set_linewidth(2.5)
            ax.set_xticks([])
            ax.set_yticks([])

    return fig


def main():
    parser = argparse.ArgumentParser(description="Visualize most-activating images per class")
    parser.add_argument('--csv_path', type=str, default=None,
                        help="Path to rankings CSV (default: rankings.csv in script dir)")
    parser.add_argument('--n_images', type=int, default=10)
    parser.add_argument('--labels', type=str, nargs='*', default=None,
                        help="Semantic labels for each class (e.g., --labels Artifacts Natural)")
    parser.add_argument('--dpi', type=int, default=600)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    imagenet_dir = os.environ.get("IMAGENET_DATA_DIR")
    if not imagenet_dir:
        raise EnvironmentError("IMAGENET_DATA_DIR not set. Source .env first.")

    csv_path = args.csv_path or os.path.join(SCRIPT_DIR, "rankings.csv")
    df = pd.read_csv(csv_path)
    n_classes = df['class_idx'].nunique()
    print(f"Loaded {len(df)} rows ({n_classes} classes) from {csv_path}")

    # Build class labels
    class_labels = {}
    if args.labels:
        for i, label in enumerate(args.labels):
            class_labels[i] = label
    else:
        for i in range(n_classes):
            class_labels[i] = f"Class {i}"

    # Load images
    images_per_class = {}
    for class_idx in sorted(df['class_idx'].unique()):
        df_c = df[df['class_idx'] == class_idx].head(args.n_images)
        print(f"  {class_labels.get(class_idx, f'Class {class_idx}')}: loading {len(df_c)} images...")
        images_per_class[class_idx] = [
            load_image(row['image_file'], imagenet_dir)
            for _, row in df_c.iterrows()
        ]

    fig = make_figure(images_per_class, class_labels, args.n_images)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    output_path = args.output or os.path.join(FIGURES_DIR, "most_activating.png")
    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
