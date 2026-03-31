"""Panel E: PCA scatter of THINGS concept representations with image insets.

Four side-by-side PC1 vs PC2 scatter plots colored by super-category:
  Behavioral | CNN 8-class CLIP | CNN 1K classes | ViT-B/16 1K classes
Each panel shows a triplet of concept images (asparagus, engine, gorilla)
connected by lines, with adaptive tile placement.
"""

import os

import numpy as np
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image

from plot_pc_scatter import (
    load_super_categories, l2_normalize, compute_pca,
    plot_scatter_panel, SUPER_ORDER, SUPER_COLORS,
)

# ── Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))

# ── Panel config ─────────────────────────────────────────────────────────
# (title, subtitle, data_key)
PC_PANELS = [
    ("Behavioral",  "(ground truth)", None),
    ("CNN",         "(8 classes (CLIP repr.))", "clip8"),
    ("CNN",         "(1K classes)",   "alexnet_pre"),
    ("ViT-B/16",    "(1K classes)",   "vit_pre"),
]

# ── Image inset config ──────────────────────────────────────────────────
THINGS_IMAGE_DIR = os.path.expanduser(
    "~/.cache/bonner-datasets/hebart2019.things/images/object_images")
INSET_CONCEPTS = ["asparagus", "engine", "gorilla"]
INSET_BORDER_COLORS = {
    "asparagus": SUPER_COLORS["Food"],
    "engine":    SUPER_COLORS["Vehicle"],
    "gorilla":   SUPER_COLORS["Animal"],
}
INSET_IMAGE_VARIANT = {
    "asparagus": "asparagus_04s.jpg",
    "engine":    "engine_08s.jpg",
}
INSET_FALLBACK_ANGLES = {
    "asparagus": np.radians(210),
    "engine":    np.radians(330),
    "gorilla":   np.radians(90),
}


def load_pc_scatter_data():
    """Load the 4 representations and concept names for PCA scatter panels.

    Returns (reps_dict, concept_names) where reps_dict maps
    data_key -> features (n_concepts, n_features).
    """
    behav_data = np.load(os.path.join(
        PROJECT_ROOT, "experiments/things_visualizations/data/things_viz_data.npz"),
        allow_pickle=True)
    activations = np.load(os.path.join(
        PROJECT_ROOT, "manuscript/figures/fig5/activations.npz"), allow_pickle=True)
    pretrained_alexnet = np.load(os.path.join(
        PROJECT_ROOT, "manuscript/figures/fig5/pretrained_alexnet_fc1.npz"))
    pretrained_vit = np.load(os.path.join(
        PROJECT_ROOT, "manuscript/figures/fig4/pretrained_vit_things.npz"))

    reps = {
        None:          behav_data["embeddings"],
        "clip8":       l2_normalize(activations["clip8_fc1"]),
        "alexnet_pre": l2_normalize(pretrained_alexnet["fc1"]),
        "vit_pre":     l2_normalize(pretrained_vit["block5"]),
    }
    return reps, list(behav_data["concept_names"])


def _load_inset_image(concept, size=256):
    """Load and resize a THINGS concept image."""
    filename = INSET_IMAGE_VARIANT.get(concept, f"{concept}_01b.jpg")
    path = os.path.join(THINGS_IMAGE_DIR, concept, filename)
    img = Image.open(path).convert("RGB").resize((size, size), Image.LANCZOS)
    return np.array(img)


def _shorten_line_to_circle(p1, p2, radius):
    """Shorten a line segment so it stops at circle edges at both endpoints."""
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    length = np.sqrt(dx**2 + dy**2)
    if length < 1e-10:
        return p1[0], p1[1], p2[0], p2[1]
    ux, uy = dx / length, dy / length
    return (p1[0] + ux * radius, p1[1] + uy * radius,
            p2[0] - ux * radius, p2[1] - uy * radius)


def _compute_tile_positions(coords, xlim, ylim):
    """Compute image tile positions using centroid-based angle placement.

    Handles angular separation, margin clamping, perpendicular sliding,
    and iterative repulsion between tiles.
    """
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    span = max(x_range, y_range)

    # Adaptive offset based on triplet spread
    centroid = np.mean([coords[c] for c in INSET_CONCEPTS], axis=0)
    triplet_spread = max(
        np.ptp([coords[c][0] for c in INSET_CONCEPTS]) / x_range,
        np.ptp([coords[c][1] for c in INSET_CONCEPTS]) / y_range,
    )
    offset_dist = (0.13 + 0.14 * min(triplet_spread / 0.4, 1.0)) * span

    # Compute "away from centroid" angles
    angles = {}
    for concept in INSET_CONCEPTS:
        away = coords[concept] - centroid
        norm = np.linalg.norm(away)
        angles[concept] = np.arctan2(away[1], away[0]) if norm > 1e-10 else INSET_FALLBACK_ANGLES[concept]

    # Enforce minimum angular separation (80 deg)
    min_sep = np.radians(80)
    concept_list = list(INSET_CONCEPTS)
    angle_list = [angles[c] for c in concept_list]
    for _ in range(20):
        changed = False
        for i in range(len(concept_list)):
            for j in range(i + 1, len(concept_list)):
                diff = (angle_list[i] - angle_list[j] + np.pi) % (2 * np.pi) - np.pi
                if abs(diff) < min_sep:
                    push = (min_sep - abs(diff)) * 0.3 * np.sign(diff)
                    angle_list[i] += push
                    angle_list[j] -= push
                    changed = True
        if not changed:
            break
    for i, c in enumerate(concept_list):
        angles[c] = angle_list[i]

    # First pass: initial positions with margin clamping
    margin_x, margin_y = 0.02 * x_range, 0.02 * y_range
    x_lo, x_hi = xlim[0] + margin_x, xlim[1] - margin_x
    y_lo, y_hi = ylim[0] + margin_y, ylim[1] - margin_y
    min_dist_circle = 0.18 * span
    min_dist_tiles = 0.12 * span

    positions = {}
    for concept in INSET_CONCEPTS:
        pt = coords[concept]
        a = angles[concept]
        ox = np.clip(pt[0] + np.cos(a) * offset_dist, x_lo, x_hi)
        oy = np.clip(pt[1] + np.sin(a) * offset_dist, y_lo, y_hi)

        # Perpendicular slide if too close to circle after clamping
        dx, dy = ox - pt[0], oy - pt[1]
        dist = np.sqrt(dx**2 + dy**2)
        if dist < min_dist_circle:
            shortfall = np.sqrt(max(min_dist_circle**2 - dist**2, 0))
            if dist > 1e-10:
                perp_x, perp_y = -dy / dist, dx / dist
            else:
                perp_x, perp_y = 1.0, 0.0
            for sign in [1, -1]:
                nx, ny = ox + sign * perp_x * shortfall, oy + sign * perp_y * shortfall
                if x_lo <= nx <= x_hi and y_lo <= ny <= y_hi:
                    ox, oy = nx, ny
                    break
            else:
                if dist > 1e-10:
                    scale = min_dist_circle / dist
                    ox, oy = pt[0] + dx * scale, pt[1] + dy * scale

        positions[concept] = [ox, oy]

    # Second pass: iterative repulsion between tiles
    for _ in range(30):
        moved = False
        for i in range(len(concept_list)):
            for j in range(i + 1, len(concept_list)):
                ci, cj = concept_list[i], concept_list[j]
                pi, pj = positions[ci], positions[cj]
                dx, dy = pi[0] - pj[0], pi[1] - pj[1]
                dist = np.sqrt(dx**2 + dy**2)
                if dist < min_dist_tiles and dist > 1e-10:
                    push = (min_dist_tiles - dist) * 0.5
                    ux, uy = dx / dist, dy / dist
                    pi[0] += ux * push; pi[1] += uy * push
                    pj[0] -= ux * push; pj[1] -= uy * push
                    moved = True
        if not moved:
            break

    # Clamp after repulsion, then shorten to 75% of distance
    for concept in INSET_CONCEPTS:
        pos = positions[concept]
        pos[0] = np.clip(pos[0], xlim[0] + margin_x, xlim[1] - margin_x)
        pos[1] = np.clip(pos[1], ylim[0] + margin_y, ylim[1] - margin_y)
        pt = coords[concept]
        pos[0] = pt[0] + (pos[0] - pt[0]) * 0.75
        pos[1] = pt[1] + (pos[1] - pt[1]) * 0.75

    return positions


def draw_image_insets(axes, all_pcs, concept_names):
    """Draw image insets for the triplet on each scatter panel.

    Places open circles at data positions, connects them with lines,
    and places images adjacent to circles with adaptive positioning.
    """
    indices = [concept_names.index(c) for c in INSET_CONCEPTS]
    images = {c: _load_inset_image(c) for c in INSET_CONCEPTS}

    for ax, pcs in zip(axes, all_pcs):
        coords = {c: pcs[idx] for c, idx in zip(INSET_CONCEPTS, indices)}
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        circle_radius = 0.025 * max(xlim[1] - xlim[0], ylim[1] - ylim[0])

        # Connecting lines between triangle vertices
        for i in range(len(INSET_CONCEPTS)):
            c1, c2 = INSET_CONCEPTS[i], INSET_CONCEPTS[(i + 1) % len(INSET_CONCEPTS)]
            x1s, y1s, x2s, y2s = _shorten_line_to_circle(
                coords[c1], coords[c2], circle_radius)
            ax.plot([x1s, x2s], [y1s, y2s],
                    color="#222222", linewidth=1.3, linestyle="-",
                    alpha=0.6, zorder=6, solid_capstyle="round")

        # Open circles at data positions
        for concept in INSET_CONCEPTS:
            x, y = coords[concept]
            ax.scatter(x, y, s=80, facecolors="none", edgecolors="black",
                       linewidths=1.5, zorder=7)

        # Compute tile positions and draw
        tile_pos = _compute_tile_positions(coords, xlim, ylim)
        for concept in INSET_CONCEPTS:
            pt = coords[concept]
            ox, oy = tile_pos[concept]
            ax.plot([pt[0], ox], [pt[1], oy],
                    color="#555555", linewidth=0.6, linestyle="-",
                    alpha=0.5, zorder=6)
            im = OffsetImage(images[concept], zoom=0.11)
            ab = AnnotationBbox(im, (ox, oy), frameon=True, pad=0.08,
                                bboxprops=dict(edgecolor="black",
                                               linewidth=0.8, facecolor="white"),
                                zorder=8)
            ax.add_artist(ab)
