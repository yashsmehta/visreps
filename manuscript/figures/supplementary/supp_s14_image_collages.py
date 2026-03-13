"""
Supplementary Figure S14: Image Collages (Concepts Where Coarse Wins vs 1000-way Wins).

Composites the existing collage images side by side with panel labels.
Uses pre-generated collages from experiments/things_visualizations/figures/.

Run from project root:
    python manuscript/figures/supplementary/supp_s14_image_collages.py
"""

import os
import sys
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "supp_s14_image_collages.png")

COLLAGE_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "..",
    "experiments", "things_visualizations", "figures"
)
COLLAGE_CLIP4_PATH = os.path.join(COLLAGE_DIR, "collage_clip4_wins.png")
COLLAGE_1K_PATH = os.path.join(COLLAGE_DIR, "collage_1k_wins.png")


def get_font(size):
    """Load a bold font at the given size, with fallbacks."""
    for path in [
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def main():
    # Check existing collages
    if not os.path.exists(COLLAGE_CLIP4_PATH):
        print(f"ERROR: Missing {COLLAGE_CLIP4_PATH}")
        print("Run: python experiments/things_visualizations/characterize_discrepant.py")
        return
    if not os.path.exists(COLLAGE_1K_PATH):
        print(f"ERROR: Missing {COLLAGE_1K_PATH}")
        print("Run: python experiments/things_visualizations/characterize_discrepant.py")
        return

    print("Loading existing collage images...")
    img_clip4 = Image.open(COLLAGE_CLIP4_PATH)
    img_1k = Image.open(COLLAGE_1K_PATH)

    # Add panel labels (a, b)
    gap = 40
    label_height = 50
    margin = 20

    # Resize to same height if needed
    max_h = max(img_clip4.height, img_1k.height)
    if img_clip4.height != max_h:
        scale = max_h / img_clip4.height
        img_clip4 = img_clip4.resize(
            (int(img_clip4.width * scale), max_h), Image.LANCZOS
        )
    if img_1k.height != max_h:
        scale = max_h / img_1k.height
        img_1k = img_1k.resize(
            (int(img_1k.width * scale), max_h), Image.LANCZOS
        )

    total_w = img_clip4.width + gap + img_1k.width + 2 * margin
    total_h = max_h + label_height + margin

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    # Panel labels
    font = get_font(32)
    x_left = margin
    x_right = margin + img_clip4.width + gap

    draw.text((x_left, 8), "A", fill=(30, 30, 30), font=font)
    draw.text((x_right, 8), "B", fill=(30, 30, 30), font=font)

    # Paste collages
    canvas.paste(img_clip4, (x_left, label_height))
    canvas.paste(img_1k, (x_right, label_height))

    canvas.save(OUTPUT_PATH, dpi=(300, 300))
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
