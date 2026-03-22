"""THINGS-behavior: RSA score across training epochs (32-way vs 1000-way).

Four-panel figure: CustomCNN, ResNet-50, ConvNeXt-Base, ViT-B/16.
"""
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

DB_PATH = "results.db"
OUTPUT_DIR = "plotters/things/figures"

# ── Style (matching reconstruction analysis) ─────────────────────────────────

COARSE_COLOR = "#2166ac"       # blue — coarse (32-way)
FINE_COLOR = "#e8963e"         # warm amber — 1000-way (matches Figure 2)
SPINE_WIDTH = 1.0
TICK_MAJOR_LEN = 4
TICK_MINOR_LEN = 2.5

RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Liberation Sans", "Nimbus Sans", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Liberation Sans",
    "mathtext.it": "Liberation Sans:italic",
}

EPOCHS = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20]
SEED = 1

MODELS = [
    {
        "model_name": "CustomCNN",
        "title": "CustomCNN",
        "queries": {
            32:   {"checkpoint_dir": "/data/ymehta3/clip_pca"},
            1000: {"checkpoint_dir": "/data/ymehta3/default"},
        },
    },
    {
        "model_name": "ResNet50",
        "title": "ResNet-50",
        "queries": {
            32:   {"checkpoint_pattern": "%resnet50%"},
            1000: {"checkpoint_pattern": "%resnet50%"},
        },
    },
    {
        "model_name": "ConvNeXt_Base",
        "title": "ConvNeXt-Base",
        "queries": {
            32:   {"checkpoint_dir": "/data/ymehta3/convnext_base_clip_pca"},
            1000: {"checkpoint_dir": "/data/ymehta3/convnext_base_default"},
        },
    },
    {
        "model_name": "ViTBase",
        "title": "ViT-B/16",
        "queries": {
            32:   {"checkpoint_pattern": "%vitbase%"},
            1000: {"checkpoint_pattern": "%vitbase%"},
        },
    },
]

sns.set_theme(style="ticks", context="paper", font_scale=1.1, rc=RC)
conn = sqlite3.connect(DB_PATH)

fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=False)

for idx, (ax, model) in enumerate(zip(axes, MODELS)):
    panel_label = chr(ord("a") + idx)

    for cfg_id, color, label, marker, ms in [
        (32, COARSE_COLOR, "32 classes (CLIP Repr.)", "s", 5.5),
        (1000, FINE_COLOR, "1,000 classes", "o", 5.5),
    ]:
        q = model["queries"][cfg_id]
        if "checkpoint_dir" in q:
            where = "checkpoint_dir = ?"
            params = [q["checkpoint_dir"]]
        else:
            where = "checkpoint_dir LIKE ?"
            params = [q["checkpoint_pattern"]]

        df = pd.read_sql(f"""
            SELECT epoch, score
            FROM results
            WHERE neural_dataset = 'things-behavior'
              AND analysis = 'rsa'
              AND compare_method = 'spearman'
              AND reconstruct_from_pcs = 0
              AND model_name = ?
              AND seed = ?
              AND cfg_id = ?
              AND {where}
            ORDER BY epoch
        """, conn, params=[model["model_name"], SEED, cfg_id] + params)
        df = df[df["epoch"].isin(EPOCHS)]

        if not df.empty:
            ax.plot(df["epoch"], df["score"], marker=marker, linestyle="-",
                    color=color, label=label, markersize=ms, linewidth=2.2,
                    markeredgecolor="white", markeredgewidth=0.7, zorder=3)

    ax.set_xlabel("Epoch", fontsize=9, labelpad=4)
    if idx == 0:
        ax.set_ylabel(r"RSA (Spearman $\rho$)", fontsize=9, labelpad=4)
    ax.set_title(f"{panel_label}.  {model['title']}", fontsize=11,
                 fontweight="bold", pad=6, loc="left")
    ax.set_xticks(EPOCHS)
    ax.tick_params(axis="both", which="major", labelsize=8,
                   length=TICK_MAJOR_LEN, width=0.8, direction="out")
    ax.yaxis.set_minor_locator(plt.matplotlib.ticker.AutoMinorLocator(2))
    ax.tick_params(axis="y", which="minor", length=TICK_MINOR_LEN,
                   width=0.6, direction="out")
    ax.yaxis.grid(True, which="major", color="#EBEBEB", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)
    sns.despine(ax=ax, right=True, top=True, offset=4)
    ax.spines["bottom"].set_linewidth(SPINE_WIDTH)
    ax.spines["left"].set_linewidth(SPINE_WIDTH)

# Share y-axis across the three ConvNet panels (a, b, c)
convnet_axes = axes[:3]
y_mins = [ax.get_ylim()[0] for ax in convnet_axes]
y_maxs = [ax.get_ylim()[1] for ax in convnet_axes]
shared_ylim = (min(y_mins), max(y_maxs))
for ax in convnet_axes:
    ax.set_ylim(shared_ylim)

# Legend inside ResNet-50 panel (bottom right)
handles, labels = axes[1].get_legend_handles_labels()
axes[1].legend(handles, labels, loc="lower right", fontsize=7.5,
               frameon=True, edgecolor="#d0d0d0", fancybox=False,
               framealpha=0.95, handlelength=2.2, labelspacing=0.35,
               borderpad=0.5, handletextpad=0.6)

plt.tight_layout(w_pad=2.5)
out = f"{OUTPUT_DIR}/things_epoch_trajectory.png"
fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
plt.close()
print(f"Saved → {out}")

conn.close()
