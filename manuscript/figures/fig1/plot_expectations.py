"""Figure 1D — Conventional expectations schematic.

Conceptual plot showing the field's default expectation: more classes should
yield better brain/behavioral alignment. Two small panels (brain, behavior)
with schematic monotonic curves and a question mark.

Usage (from project root):
    python manuscript/figures/fig1/plot_expectations.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

sns.set_theme(style="ticks", context="paper", font_scale=1.05)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.linewidth": 1.0,
})


def draw_expectation_panel(ax, title, icon_text, color):
    """Draw one schematic expectation panel."""
    # X positions: log-spaced class counts
    x_ticks = [2, 4, 8, 16, 32, 64, 1000]
    x_log = np.log2(x_ticks)

    # Schematic curve (no real data — purely conceptual)
    x_smooth = np.linspace(x_log[0], x_log[-1], 200)

    # Smooth monotonic increase (saturating log-like)
    t = (x_smooth - x_smooth[0]) / (x_smooth[-1] - x_smooth[0])
    y = 0.15 + 0.55 * (1 - np.exp(-3.5 * t))

    ax.plot(x_smooth, y, color=color, linewidth=2.5, solid_capstyle="round",
            zorder=3)
    # Shaded uncertainty region
    ax.fill_between(x_smooth, y - 0.04, y + 0.04, color=color, alpha=0.10,
                    zorder=2)

    # Question mark annotation
    ax.text(0.50, 0.92, "?", transform=ax.transAxes, fontsize=28,
            fontweight="bold", color="#888888", ha="center", va="top",
            alpha=0.6)

    # Icon text (brain or behavior) in top-left
    ax.text(0.05, 0.92, icon_text, transform=ax.transAxes, fontsize=9,
            fontweight="bold", color=color, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.12,
                      edgecolor=color, linewidth=0.8))

    # Axis formatting
    ax.set_xticks(x_log)
    ax.set_xticklabels([str(x) for x in x_ticks], fontsize=8.5)
    ax.set_xlabel("Number of Classes", fontsize=10, labelpad=4)
    ax.set_ylabel("Alignment", fontsize=10, labelpad=4)
    ax.set_title(title, fontsize=11, fontweight="semibold", pad=8)

    ax.set_ylim(0.05, 0.85)
    ax.set_yticks([])  # No real y-values — this is schematic
    ax.tick_params(axis="x", length=4, width=0.8)

    sns.despine(ax=ax, left=True, offset=4)
    ax.spines["bottom"].set_linewidth(1.0)


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.2))

    draw_expectation_panel(
        ax1, "Neural Alignment", "Brain",
        color="#2166AC",
    )
    draw_expectation_panel(
        ax2, "Behavioral Alignment", "Behavior",
        color="#B2182B",
    )

    plt.tight_layout(w_pad=3)
    out = os.path.join(SCRIPT_DIR, "expectations.png")
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white",
                edgecolor="none")
    print(f"Saved -> {out}")
    plt.close()


if __name__ == "__main__":
    main()
