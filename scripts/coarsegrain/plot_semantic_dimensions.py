"""Visual sanity check of the hand-assigned semantic dimensions.

For each dimension, build one montage: left panel = classes scored 0, right panel =
classes scored 1. One thumbnail per ImageNet class (1,000 total per figure), with the
class name captioned under each thumbnail so misassignments are easy to spot.

Usage (from project root, .env loaded):
    python scripts/coarsegrain/plot_semantic_dimensions.py --dims natural handheld indoor

Writes ``pca_labels/pca_labels_semantic/figures/dim_{name}.jpg``.
"""
import argparse
import os
import random

import pandas as pd
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

DIMS_PATH = "pca_labels/pca_labels_semantic/class_dimensions.csv"
OUT_DIR = "pca_labels/pca_labels_semantic/figures"

THUMB = 224       # thumbnail side in px (ImageNet native eval resolution)
CAPTION_H = 26    # caption strip under each thumbnail
COLS = 25         # thumbnails per row in each panel
GAP = 120         # gap between the two panels
TITLE_H = 120


def pick_image(class_dir, rng):
    files = sorted(f for f in os.listdir(class_dir) if f.upper().endswith(".JPEG"))
    return os.path.join(class_dir, rng.choice(files))


def load_thumb(path):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    side = min(w, h)  # center crop to square, then resize
    left, top = (w - side) // 2, (h - side) // 2
    return img.crop((left, top, left + side, top + side)).resize((THUMB, THUMB), Image.LANCZOS)


def build_panel(rows, font):
    """rows: list of (thumb_path, class_name). Returns a PIL image grid."""
    n_rows = (len(rows) + COLS - 1) // COLS
    cell_h = THUMB + CAPTION_H
    panel = Image.new("RGB", (COLS * THUMB, n_rows * cell_h), "white")
    draw = ImageDraw.Draw(panel)
    for i, (path, name) in enumerate(rows):
        r, c = divmod(i, COLS)
        x, y = c * THUMB, r * cell_h
        panel.paste(load_thumb(path), (x, y))
        draw.text((x + 4, y + THUMB + 3), name[:30], fill="black", font=font)
    return panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", nargs="+", default=["natural", "handheld", "indoor"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    load_dotenv()
    root = os.environ["IMAGENET_DATA_DIR"]
    dims = pd.read_csv(DIMS_PATH)
    os.makedirs(OUT_DIR, exist_ok=True)

    # One image per class, shared across all three figures so they are comparable.
    rng = random.Random(args.seed)
    dims["image"] = [pick_image(os.path.join(root, w), rng) for w in dims.wnid]

    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 17)
        title_font = ImageFont.truetype("DejaVuSans-Bold.ttf", 64)
    except OSError:
        font = title_font = ImageFont.load_default()

    for dim in args.dims:
        panels = []
        for value in (0, 1):
            sub = dims[dims[dim] == value]
            panels.append((value, len(sub), build_panel(list(zip(sub.image, sub.class_name)), font)))

        width = sum(p.width for _, _, p in panels) + GAP
        height = TITLE_H + max(p.height for _, _, p in panels)
        fig = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(fig)
        x = 0
        for value, n, panel in panels:
            draw.text((x, 25), f"{dim} = {value}   ({n} classes)", fill="black", font=title_font)
            fig.paste(panel, (x, TITLE_H))
            x += panel.width + GAP

        out = os.path.join(OUT_DIR, f"dim_{dim}.jpg")
        fig.save(out, quality=90)
        print(f"saved {out}  ({width}x{height})")


if __name__ == "__main__":
    main()
