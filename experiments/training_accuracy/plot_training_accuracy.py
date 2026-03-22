#!/usr/bin/env python3
"""Plot CustomCNN test accuracy by label granularity and source."""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import numpy as np

# ── Style ──────────────────────────────────────────────────────────────
sns.set_theme(style="ticks", context="paper", font_scale=1.2)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
})

# ── Data ───────────────────────────────────────────────────────────────
df = pd.read_csv("/home/ymehta3/research/VisionAI/visreps/logs/training_accuracy.csv")
df = df[(df["model_name"] == "CustomCNN") &
        (df["label_source"].isin(["alexnet", "clip", "vit", "dino"]))].copy()

levels = [2, 4, 8, 16, 32, 64]
source_order = ["alexnet", "clip", "vit", "dino"]

# Four shades of blue
colors = {
    "alexnet": "#08519c",
    "clip":    "#3182bd",
    "vit":     "#6baed6",
    "dino":    "#bdd7e7",
}
display_names = {"alexnet": "AlexNet", "clip": "CLIP", "vit": "ViT", "dino": "DINO"}


def rounded_bar(ax, x, height, width, color, bottom=0, radius=0.12):
    """Draw a bar with rounded top corners using FancyBboxPatch.

    Extends 2*radius below `bottom` so the rounded bottom corners are
    clipped by the axis limits, giving the appearance of flat bottoms.
    """
    overshoot = 2 * radius
    rect = FancyBboxPatch(
        (x - width / 2, bottom - overshoot), width, height + overshoot,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=color, edgecolor="white", linewidth=0.5,
        zorder=2, clip_on=True,
    )
    ax.add_patch(rect)
    return rect


# ── Figure ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(9, 5.5), sharey=True)
axes_flat = axes.flatten()

bar_width = 0.55

for idx, n in enumerate(levels):
    ax = axes_flat[idx]
    sub = df[df["n_classes"] == n]

    means, sems = [], []
    for src in source_order:
        vals = sub[sub["label_source"] == src]["test_acc"].values
        means.append(vals.mean())
        sems.append(vals.std() / np.sqrt(len(vals)) if len(vals) > 1 else 0)

    x = np.arange(len(source_order))

    # Rounded bars
    for i, (src, m) in enumerate(zip(source_order, means)):
        rounded_bar(ax, x[i], m - 30, bar_width, colors[src], bottom=30)

    # Error bars
    ax.errorbar(x, means, yerr=sems, fmt="none", ecolor="#333333",
                elinewidth=0.8, capsize=2.5, capthick=0.8, zorder=4)

    # Seed dots
    for i, src in enumerate(source_order):
        seed_vals = sub[sub["label_source"] == src]["test_acc"].values
        rng = np.random.default_rng(42)
        jitter = rng.uniform(-0.06, 0.06, size=len(seed_vals))
        ax.scatter(np.full_like(seed_vals, i) + jitter, seed_vals,
                   color="#1a1a1a", s=14, zorder=5, alpha=0.5,
                   linewidths=0.3, edgecolors="white")

    # Chance level
    chance = 100.0 / n
    if chance >= 30:
        ax.axhline(chance, color="#aaaaaa", linestyle="--", linewidth=0.5,
                   alpha=0.6, zorder=1)
        ax.text(len(source_order) - 0.55, chance + 0.8, "chance",
                fontsize=6, color="#999999", ha="right", va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels([display_names[s] for s in source_order], fontsize=8)

    # Panel letter + title
    panel_letter = chr(ord("a") + idx)
    ax.set_title(f"{n}-way", fontsize=9.5, fontweight="semibold", pad=6)
    ax.text(-0.08, 1.08, panel_letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="right")

    # Despine with offset
    sns.despine(ax=ax, offset=5)
    ax.set_ylim(28, 102)
    ax.set_xlim(-0.5, len(source_order) - 0.5)

    # Subtle horizontal gridlines
    ax.yaxis.grid(True, linewidth=0.3, alpha=0.3, color="#cccccc")
    ax.set_axisbelow(True)

    if idx % 3 == 0:
        ax.set_ylabel("Test accuracy (%)", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)

plt.tight_layout(h_pad=1.5, w_pad=1.0)
plt.savefig("/home/ymehta3/research/VisionAI/visreps/logs/training_accuracy.png",
            dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none")
print("Saved")
