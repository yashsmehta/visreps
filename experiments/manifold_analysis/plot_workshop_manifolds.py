"""Plot the SNR-only manifold figure used by the NeurReps extended abstract."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, FuncFormatter

HELDOUT_RESULTS = Path("experiments/manifold_analysis/heldout_sweep_results/results.json")
OUTPUT = Path("manuscript/NeurReps_2026/figure3.png")
COARSE = (2, 4, 8, 16, 32, 64)
SEEDS = "abc"
COARSE_COLOR = "#1769aa"
BASELINE_COLOR = "#d97921"


def values(results: dict) -> tuple[np.ndarray, np.ndarray]:
    coarse = np.asarray([
        [results[f"cfg{n}{seed}"]["snr"] for seed in SEEDS]
        for n in COARSE
    ])
    baseline = np.asarray([
        results[f"cfg1000{seed}"]["snr"] for seed in SEEDS
    ])
    return coarse, baseline


def main() -> None:
    results = json.loads(HELDOUT_RESULTS.read_text())
    coarse, baseline = values(results)
    means = coarse.mean(axis=1)
    positions = np.arange(len(COARSE), dtype=float)
    baseline_x = 7.15

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.linewidth": 0.8,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, ax = plt.subplots(figsize=(5.25, 2.75))

    ax.errorbar(
        positions, means,
        yerr=[means - coarse.min(axis=1), coarse.max(axis=1) - means],
        fmt="o-", markersize=5.5, color=COARSE_COLOR, linewidth=1.7,
        markeredgecolor="white", markeredgewidth=0.7,
        elinewidth=1.1, capsize=3, capthick=1.1, zorder=3,
    )

    baseline_mean = baseline.mean()
    ax.errorbar(
        baseline_x, baseline_mean,
        yerr=[[baseline_mean - baseline.min()], [baseline.max() - baseline_mean]],
        fmt="D", markersize=6.2, color=BASELINE_COLOR,
        markeredgecolor="white", markeredgewidth=0.7,
        elinewidth=1.1, capsize=3, capthick=1.1, zorder=4,
    )

    ax.set_xticks([*positions, baseline_x], [*map(str, COARSE), "1,000"])
    ax.set_xlim(-0.42, baseline_x + 0.48)
    ax.set_xlabel("Number of training categories", labelpad=7)
    ax.set_ylabel("Manifold SNR", labelpad=7)
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda value, _: f"{value:.3f}".rstrip("0").rstrip(".")))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="both", which="major", direction="out", length=3.5)
    ax.tick_params(axis="y", which="minor", direction="out", length=2, width=0.4)
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.55, zorder=0)
    ax.margins(y=0.16)

    break_x = 6.10
    for offset in (-0.03, 0.11):
        ax.plot(
            [break_x + offset - 0.05, break_x + offset + 0.05],
            [-0.018, 0.018], transform=ax.get_xaxis_transform(),
            color="#333333", lw=0.8, clip_on=False,
        )

    fig.tight_layout(pad=0.7)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
