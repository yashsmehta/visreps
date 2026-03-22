"""THINGS-behavior: RSA bar plot comparison at epoch 20 (32-way vs 1000-way).

Grouped bar plot: CustomCNN, ResNet-50, ConvNeXt-Base, ViT-B/16.
"""
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch

DB_PATH = "results.db"
OUTPUT_DIR = "plotters/things/figures"

# ── Style ────────────────────────────────────────────────────────────────────
COARSE_COLOR = "#2166ac"       # blue — coarse (32-way)
FINE_COLOR = "#e8963e"         # warm amber — 1000-way (matches Figure 2)
SPINE_WIDTH = 1.0
TICK_MAJOR_LEN = 4

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Nimbus Sans", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans",
    "mathtext.it": "Liberation Sans:italic",
}

SEED = 1
EPOCH = 20

MODELS = [
    {
        "model_name": "CustomCNN",
        "label": "CustomCNN",
        "queries": {
            32:   {"checkpoint_dir": "/data/ymehta3/clip_pca"},
            1000: {"checkpoint_dir": "/data/ymehta3/default"},
        },
    },
    {
        "model_name": "ResNet50",
        "label": "ResNet-50",
        "queries": {
            32:   {"checkpoint_pattern": "%resnet50%"},
            1000: {"checkpoint_pattern": "%resnet50%"},
        },
    },
    {
        "model_name": "ConvNeXt_Base",
        "label": "ConvNeXt-Base",
        "queries": {
            32:   {"checkpoint_dir": "/data/ymehta3/convnext_base_clip_pca"},
            1000: {"checkpoint_dir": "/data/ymehta3/convnext_base_default"},
        },
    },
    {
        "model_name": "ViTBase",
        "label": "ViT-B/16",
        "queries": {
            32:   {"checkpoint_pattern": "%vitbase%"},
            1000: {"checkpoint_pattern": "%vitbase%"},
        },
    },
]

sns.set_theme(style="ticks", context="paper", font_scale=1.1, rc=RC)
conn = sqlite3.connect(DB_PATH)

# ── Fetch scores ─────────────────────────────────────────────────────────────
scores = {}  # {model_label: {cfg_id: score}}
for model in MODELS:
    scores[model["label"]] = {}
    for cfg_id in [1000, 32]:
        q = model["queries"][cfg_id]
        if "checkpoint_dir" in q:
            where = "checkpoint_dir = ?"
            params = [q["checkpoint_dir"]]
        else:
            where = "checkpoint_dir LIKE ?"
            params = [q["checkpoint_pattern"]]

        df = pd.read_sql(f"""
            SELECT score FROM results
            WHERE neural_dataset = 'things-behavior'
              AND analysis = 'rsa'
              AND compare_method = 'spearman'
              AND reconstruct_from_pcs = 0
              AND model_name = ?
              AND seed = ?
              AND epoch = ?
              AND cfg_id = ?
              AND {where}
            ORDER BY score DESC
            LIMIT 1
        """, conn, params=[model["model_name"], SEED, EPOCH, cfg_id] + params)

        scores[model["label"]][cfg_id] = df["score"].iloc[0] if not df.empty else 0.0

conn.close()

# ── Plot ─────────────────────────────────────────────────────────────────────
Y_MIN, Y_MAX = 0.2, 0.6
fig, ax = plt.subplots(figsize=(5.5, 4))

model_labels = [m["label"] for m in MODELS]
x = np.arange(len(model_labels))
bar_width = 0.30

conditions = [
    (1000, FINE_COLOR, "1,000 classes", -bar_width / 2),
    (32, COARSE_COLOR, "32 classes (CLIP Repr.)", bar_width / 2),
]

for cfg_id, color, label, offset in conditions:
    vals = [scores[m][cfg_id] for m in model_labels]
    # Bars start from Y_MIN (truncated axis)
    heights = [v - Y_MIN for v in vals]
    bars = ax.bar(x + offset, heights, bar_width, bottom=Y_MIN,
                  color=color, label=label,
                  edgecolor="white", linewidth=0.6, zorder=3)

    # Round the bar corners
    for bar in bars:
        bar.set_visible(False)
        fancy = FancyBboxPatch(
            (bar.get_x(), bar.get_y()),
            bar.get_width(), bar.get_height(),
            boxstyle="round,pad=0,rounding_size=0.008",
            facecolor=color, edgecolor="white", linewidth=0.6, zorder=3,
        )
        ax.add_patch(fancy)

    # Score annotations above bars
    for i, v in enumerate(vals):
        ax.text(x[i] + offset, v + 0.006, f".{int(round(v*1000)) % 1000:03d}",
                ha="center", va="bottom", fontsize=7, color="#444444")

# ── Formatting ───────────────────────────────────────────────────────────────
ax.set_ylabel(r"RSA (Spearman $\rho$)", fontsize=10, labelpad=6)
ax.set_xticks(x)
ax.set_xticklabels(model_labels, fontsize=9)
ax.tick_params(axis="y", which="major", labelsize=8.5,
               length=TICK_MAJOR_LEN, width=0.8, direction="out")
ax.tick_params(axis="x", which="major", length=0)
ax.yaxis.set_minor_locator(plt.matplotlib.ticker.AutoMinorLocator(2))
ax.tick_params(axis="y", which="minor", length=2.5, width=0.6, direction="out")
ax.yaxis.grid(True, which="major", color="#EBEBEB", linewidth=0.4, zorder=0)
ax.set_axisbelow(True)
ax.set_ylim(Y_MIN, Y_MAX)
sns.despine(ax=ax, right=True, top=True, offset=4)
ax.spines["bottom"].set_linewidth(SPINE_WIDTH)
ax.spines["left"].set_linewidth(SPINE_WIDTH)

ax.legend(loc="upper left", fontsize=7.5, frameon=True, edgecolor="#d0d0d0",
          fancybox=False, framealpha=0.95, handlelength=1.8, labelspacing=0.35,
          borderpad=0.5)

plt.tight_layout()
out = f"{OUTPUT_DIR}/things_epoch_barplot.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
plt.close()
print(f"Saved → {out}")
