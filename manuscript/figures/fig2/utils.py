"""Shared utilities for Figure 2 panels."""

import os

import numpy as np
from PIL import Image


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../.."))

# ── Data paths ──────────────────────────────────────────────────────────
DATA_4WAY = os.path.join(PROJECT_ROOT, "experiments", "representation_analysis",
                         "2pcs_compare", "data_4way_alexnet.npz")

# ── Colors ──────────────────────────────────────────────────────────────
REPR_COLORS_4 = ["#1B7A4F", "#50C888", "#E88A2A", "#D63540"]

INSET_LAYER = "fc1"


# ── Image helpers ──────────────────────────────────────────────────────

_thumb_cache = {}


def get_thumbnail(path, size=48):
    if path not in _thumb_cache:
        try:
            img = Image.open(path).convert("RGB")
            img = img.resize((size, size), Image.LANCZOS)
            _thumb_cache[path] = np.array(img)
        except Exception:
            _thumb_cache[path] = None
    return _thumb_cache[path]


# ── PCA alignment ────────────────────────────────────────────────────

def discrete_align_pcs(pcs_to_align, pcs_reference, labels, n_classes):
    """Align via optimal sign flips to match quadrant arrangement."""
    centroids_ref = np.array([pcs_reference[labels == c].mean(axis=0)
                              for c in range(n_classes)])
    cr = centroids_ref - centroids_ref.mean(axis=0)
    cr /= np.maximum(cr.std(axis=0), 1e-8)

    best_flips, best_cost = (1, 1), np.inf
    for s1 in [1, -1]:
        for s2 in [1, -1]:
            flipped = pcs_to_align * np.array([s1, s2])
            ca = np.array([flipped[labels == c].mean(axis=0) for c in range(n_classes)])
            ca = ca - ca.mean(axis=0)
            ca /= np.maximum(ca.std(axis=0), 1e-8)
            cost = np.sum((ca - cr) ** 2)
            if cost < best_cost:
                best_cost, best_flips = cost, (s1, s2)

    print(f"  Discrete alignment: flips = {best_flips}, cost = {best_cost:.4f}")
    return pcs_to_align * np.array(best_flips)


# ── Style setup ───────────────────────────────────────────────────────

def setup_style():
    """Configure matplotlib style for Figure 2."""
    import seaborn as sns
    import matplotlib.pyplot as plt
    sns.set_theme(style="ticks", context="paper", font_scale=1.1)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
