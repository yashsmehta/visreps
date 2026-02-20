"""Publication-quality k-fold RSA fluctuation plot (2x2 grid).

Shows per-fold RSA scores on the select (20%) and eval (80%) sets,
with SD reported as a percentage (coefficient of variation).
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ── Style ────────────────────────────────────────────────────────────────
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

SEL_COLOR  = "#4C72B0"
EVAL_COLOR = "#DD8452"
REF_COLOR  = "#2d2d2d"

# ── Data ─────────────────────────────────────────────────────────────────
with open("experiments/stimulus_sensitivity/data.json") as f:
    all_data = json.load(f)

ROW_CONFIGS = [
    ("conv4_early visual stream",    "conv4 / Early Visual"),
    ("fc1_ventral visual stream",    "fc1 / Ventral Visual"),
]
COL_METHODS = ["Spearman", "Kendall"]

# ── Figure ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(8, 7))

for row_i, (data_key, row_label) in enumerate(ROW_CONFIGS):
    entry = all_data[data_key]
    n_stimuli = entry["n_stimuli"]
    n_folds = entry["n_folds"]

    for col_i, cmp in enumerate(COL_METHODS):
        ax = axes[row_i, col_i]
        r = entry["results"][cmp]
        select = np.array(r["select"])
        evl    = np.array(r["eval"])
        full   = r["full"]

        # ── Jittered strip points ────────────────────────────────────
        rng = np.random.default_rng(0)
        jitter_w = 0.05
        sel_x = rng.normal(0, jitter_w, len(select))
        evl_x = rng.normal(1, jitter_w, len(evl))

        ax.scatter(
            sel_x, select, s=55, alpha=0.85, color=SEL_COLOR,
            edgecolors="white", linewidths=0.6, zorder=4,
            label=f"Select (20%)",
        )
        ax.scatter(
            evl_x, evl, s=55, alpha=0.85, color=EVAL_COLOR,
            edgecolors="white", linewidths=0.6, zorder=4,
            label=f"Eval (80%)",
        )

        # ── Mean bars ────────────────────────────────────────────────
        bar_hw = 0.15
        for xc, vals, col in [(0, select, SEL_COLOR), (1, evl, EVAL_COLOR)]:
            m = vals.mean()
            ax.plot(
                [xc - bar_hw, xc + bar_hw], [m, m],
                color=col, linewidth=2.4, zorder=5, solid_capstyle="round",
            )

        # ── Full-set reference line ──────────────────────────────────
        ax.axhline(
            full, color=REF_COLOR, ls="--", lw=1.2, alpha=0.55, zorder=2,
            label=f"Full set = {full:.4f}",
        )

        # ── CV% annotations under each group ────────────────────────
        sel_cv = (select.std() / select.mean()) * 100
        evl_cv = (evl.std() / evl.mean()) * 100

        all_vals = np.concatenate([select, evl])
        y_bot = all_vals.min()

        ax.text(
            0, y_bot - 0.012, f"CV = {sel_cv:.1f}%",
            ha="center", va="top", fontsize=9, color=SEL_COLOR, fontweight="bold",
        )
        ax.text(
            1, y_bot - 0.012, f"CV = {evl_cv:.1f}%",
            ha="center", va="top", fontsize=9, color=EVAL_COLOR, fontweight="bold",
        )

        # ── Axes formatting ──────────────────────────────────────────
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Select (20%)", "Eval (80%)"], fontsize=12)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.tick_params(axis="y", labelsize=13)
        y_top = max(all_vals.max(), full)
        ax.set_ylim(0, y_top + 0.04)
        ax.set_xlim(-0.45, 1.45)

        # Column title on top row only
        if row_i == 0:
            ax.set_title(f"Pearson \u2192 {cmp}", fontsize=15, fontweight="bold", pad=12)

        # Row label on left column only
        if col_i == 0:
            ax.set_ylabel(f"{row_label}\nRSA score", fontsize=13)
        else:
            ax.set_ylabel("")

        # Legend
        leg = ax.legend(
            fontsize=8.5, loc="lower right",
            frameon=True, framealpha=0.95, edgecolor="black", fancybox=False,
            handletextpad=0.4, borderpad=0.5,
        )
        leg.get_frame().set_linewidth(0.8)

sns.despine(fig=fig, right=True, top=True, offset=8)

fig.suptitle(
    f"RSA Fluctuation Across {n_folds}-Fold CV\n"
    f"(1000-way AlexNet, NSD subject 0, {n_stimuli} stimuli)",
    fontsize=14, fontweight="bold", y=1.02,
)

plt.tight_layout()
fig.savefig(
    "experiments/stimulus_sensitivity/stimulus_sensitivity.png",
    dpi=600, bbox_inches="tight", facecolor="white", edgecolor="none",
)
print("Saved experiments/stimulus_sensitivity/stimulus_sensitivity.png")
