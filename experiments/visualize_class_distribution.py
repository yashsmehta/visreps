"""Visualize class distribution from PCA labels."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent

# ============ CONFIGURATION ============
LABELS_PATH = SCRIPT_DIR / "../pca_labels/pca_labels_alexnet/n_classes_4096.csv"
OUTPUT_PATH = SCRIPT_DIR / "results/class_distribution.png"
N_SHOW = 16  # Number of top/bottom classes to show
# =======================================

# Load and count
df = pd.read_csv(LABELS_PATH)
class_counts_series = df["pca_label"].value_counts().sort_values(ascending=False)
class_counts = class_counts_series.values
n_classes = len(class_counts)

n_show = min(N_SHOW, n_classes // 2)  # Ensure we don't overlap top/bottom
top_n = class_counts_series.head(n_show)
bottom_n = class_counts_series.tail(n_show).sort_values(ascending=False)

# Style
plt.style.use("seaborn-v0_8-whitegrid")
fig, axes = plt.subplots(1, 3, figsize=(16, 5), gridspec_kw={"width_ratios": [1.2, 1, 1]})

# --- Left: Distribution histogram ---
ax = axes[0]
log_min = np.floor(np.log10(max(class_counts.min(), 1)))
log_max = np.ceil(np.log10(class_counts.max()))
bins = np.logspace(log_min, log_max, 25)
ax.hist(class_counts, bins=bins, edgecolor="white", linewidth=0.8, alpha=0.9, color="#6b7280")

ax.set_xscale("log")
ax.set_xlabel("Images per class", fontsize=11)
ax.set_ylabel("Number of classes", fontsize=11)
ax.set_title("Class Size Distribution", fontsize=12, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --- Middle: Top N classes ---
ax = axes[1]
colors_top = plt.cm.Oranges(np.linspace(0.4, 0.9, n_show))[::-1]
ax.bar(range(n_show), top_n.values, color=colors_top, edgecolor="white", linewidth=0.5)
ax.set_xlabel("Rank", fontsize=11)
ax.set_ylabel("Number of images", fontsize=11)
ax.set_title(f"Top {n_show} Classes", fontsize=12, fontweight="bold")
ax.set_xticks([0, n_show // 2, n_show - 1])
ax.set_xticklabels(["1", str(n_show // 2 + 1), str(n_show)])
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --- Right: Bottom N classes ---
ax = axes[2]
colors_bot = plt.cm.Blues(np.linspace(0.4, 0.9, n_show))[::-1]
ax.bar(range(n_show), bottom_n.values, color=colors_bot, edgecolor="white", linewidth=0.5)
ax.set_xlabel("Rank", fontsize=11)
ax.set_ylabel("Number of images", fontsize=11)
ax.set_title(f"Bottom {n_show} Classes", fontsize=12, fontweight="bold")
ax.set_xticks([0, n_show // 2, n_show - 1])
ax.set_xticklabels([str(n_classes - n_show + 1), str(n_classes - n_show // 2), str(n_classes)])
ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Suptitle
fig.suptitle(
    f"{len(class_counts):,} classes  ·  {len(df):,} images  ·  Median: {np.median(class_counts):.0f}  ·  Range: {class_counts.min()}–{class_counts.max()}",
    fontsize=10, color="#555", y=0.02
)

plt.tight_layout(rect=[0, 0.05, 1, 1])
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=150, facecolor="white", bbox_inches="tight")
plt.show()

print(f"Saved to {OUTPUT_PATH}")
