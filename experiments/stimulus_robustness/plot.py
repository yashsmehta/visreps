"""Publication-quality 2x4 plot: stimulus robustness of coarse-grain
brain alignment across network layers.

Layout:
  Rows:    early visual stream (top), ventral visual stream (bottom)
  Columns: AlexNet-PCA, ViT-PCA, CLIP-PCA, DINO-PCA

Each panel shows 3 lines for the same model:
  - All stimuli (~907)
  - Train half (50%)
  - Test half (50%)

Plus the AlexNet-1K baseline (all stimuli) as a dashed reference.

Usage:
    source /home/ymehta3/research/VisionAI/visreps/.venv/bin/activate && \
    python experiments/stimulus_robustness/plot.py
"""

import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ── Style ─────────────────────────────────────────────────────────────────
sns.set_theme(style="ticks", context="paper", font_scale=1.2)
plt.rcParams.update({
    "axes.linewidth": 1.8,
    "xtick.major.width": 1.4,
    "ytick.major.width": 1.4,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
})

# ── Line styles for the 3 conditions + baseline ──────────────────────────
# Same marker shape ("o") for all conditions; color differentiates them.
# Pre-activation = hollow (outline), post-activation = filled (solid).
CONDITION_STYLE = {
    "all_stimuli": {"color": "#2d2d2d", "ls": "-",  "lw": 2.2, "ms": 5,
                    "label": "All stimuli"},
    "train_50":    {"color": "#4C72B0", "ls": "-",  "lw": 1.8, "ms": 5,
                    "label": "Train 50%"},
    "test_50":     {"color": "#DD8452", "ls": "-",  "lw": 1.8, "ms": 5,
                    "label": "Test 50%"},
}
BASELINE_STYLE = {"color": "#999999", "ls": "--", "lw": 1.5,
                  "label": "AlexNet-1K (all)"}

# ── Layout ────────────────────────────────────────────────────────────────
BASE_LAYERS = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2"]
DELTA = 0.15

REGIONS = ["early visual stream", "ventral visual stream"]
REGION_SHORT = {
    "early visual stream": "Early Visual",
    "ventral visual stream": "Ventral Visual",
}

# Column order: the 4 coarse-grain PCA models
PCA_MODELS = {
    "early visual stream":  ["AlexNet-PCA-64", "ViT-PCA-16",  "CLIP-PCA-16",  "DINO-PCA-16"],
    "ventral visual stream": ["AlexNet-PCA-64", "ViT-PCA-64", "CLIP-PCA-64",  "DINO-PCA-64"],
}
COL_TITLES = ["AlexNet-PCA", "ViT-PCA", "CLIP-PCA", "DINO-PCA"]


# ── Helpers ───────────────────────────────────────────────────────────────
def layer_x_positions(layer_names):
    """Map layer names to x-coordinates with pre/post offsets."""
    pos = {}
    for i, base in enumerate(BASE_LAYERS):
        center = i + 1
        pre, post = f"{base}_pre", f"{base}_post"
        if pre in layer_names and post in layer_names:
            pos[pre] = center - DELTA
            pos[post] = center + DELTA
        elif base in layer_names:
            pos[base] = center
    return pos


def plot_line(ax, layers, scores, xpos, style, is_baseline=False):
    """Plot a condition line with hollow pre-activation and filled post-activation markers.

    Baseline lines are drawn without markers (dashed reference).
    """
    ordered = sorted(xpos.keys(), key=lambda l: xpos[l])
    xs = [xpos[l] for l in ordered]
    ys = [scores[l] for l in ordered]
    color = style["color"]

    # Draw the connecting line (no markers)
    ax.plot(
        xs, ys,
        color=color, linestyle=style["ls"], linewidth=style["lw"],
        marker=None, label=style["label"], alpha=0.85, zorder=3,
    )

    if is_baseline:
        return

    ms = style["ms"]
    # Separate pre and post points
    pre_xs  = [xpos[l] for l in ordered if l.endswith("_pre")]
    pre_ys  = [scores[l] for l in ordered if l.endswith("_pre")]
    post_xs = [xpos[l] for l in ordered if l.endswith("_post")]
    post_ys = [scores[l] for l in ordered if l.endswith("_post")]

    # Hollow circles for pre-activation
    ax.scatter(pre_xs, pre_ys, s=ms**2, marker="o",
               facecolors="white", edgecolors=color, linewidths=1.2,
               alpha=0.9, zorder=4)
    # Filled circles for post-activation
    ax.scatter(post_xs, post_ys, s=ms**2, marker="o",
               facecolors=color, edgecolors=color, linewidths=0.6,
               alpha=0.9, zorder=4)


# ── Load data ─────────────────────────────────────────────────────────────
with open("experiments/stimulus_robustness/data.json") as f:
    data = json.load(f)

# ── Figure ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 7), sharey="row")

for row_i, region in enumerate(REGIONS):
    region_data = data[region]

    # Get AlexNet-1K baseline for this region
    baseline = region_data["AlexNet-1K"]
    baseline_layers = baseline["layer_order"]
    baseline_xpos = layer_x_positions(baseline_layers)

    for col_i, model_name in enumerate(PCA_MODELS[region]):
        ax = axes[row_i, col_i]
        mdata = region_data[model_name]
        layers = mdata["layer_order"]
        xpos = layer_x_positions(layers)

        # Baseline reference (dashed, no markers)
        plot_line(ax, baseline_layers, baseline["all_stimuli"], baseline_xpos, BASELINE_STYLE,
                  is_baseline=True)

        # 3 conditions for this coarse model
        for cond_key, style in CONDITION_STYLE.items():
            plot_line(ax, layers, mdata[cond_key], xpos, style)

        # ── Axes formatting ───────────────────────────────────────
        ax.set_xticks(range(1, len(BASE_LAYERS) + 1))
        ax.set_xticklabels(BASE_LAYERS, fontsize=9, rotation=45, ha="right")
        ax.set_xlim(0.5, len(BASE_LAYERS) + 0.5)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.tick_params(axis="both", labelsize=9)

        # Column title on top row
        if row_i == 0:
            ax.set_title(COL_TITLES[col_i], fontsize=13, fontweight="bold", pad=10)

        # Row label on left column
        if col_i == 0:
            ax.set_ylabel(f"{REGION_SHORT[region]}\nSpearman RSA", fontsize=11)

        # Legend in rightmost column only
        if col_i == 3:
            leg = ax.legend(
                fontsize=7.5, loc="best",
                frameon=True, framealpha=0.9, edgecolor="gray", fancybox=False,
            )
            leg.get_frame().set_linewidth(0.6)

sns.despine(fig=fig, right=True, top=True, offset=8)

fig.suptitle(
    "Stimulus Robustness: Layer-wise RSA Under Stimulus Subsampling\n"
    "(NSD shared stimuli, subject 0)",
    fontsize=14, fontweight="bold", y=1.02,
)

plt.tight_layout()
fig.savefig(
    "experiments/stimulus_robustness/stimulus_robustness.png",
    dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none",
)
print("Saved experiments/stimulus_robustness/stimulus_robustness.png")
