"""Supplementary Figure S7: Training Summary -- Test accuracy vs number of classes.

Single panel: x-axis = number of classes (log scale), y-axis = final test accuracy
at epoch 20. Mean +/- SEM across 3 seeds. Connected line plot with markers.

Usage:
    python manuscript/figures/supplementary/supp_s7_training_summary.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator, AutoMinorLocator
import seaborn as sns

sys.path.insert(0, ".")
from manuscript.figures.fig_utils import setup_style, GRAN_COLORS, GRAN_MARKERS, COARSE_CFGS

OUTPUT = "manuscript/figures/supplementary/supp_s7_training_summary.png"

GRAN_CFGS_ALL = COARSE_CFGS + [1000]
SEED_LETTERS = ["a", "b", "c"]
FINAL_EPOCH = 20


def get_csv_path(cfg_id, seed_letter):
    if cfg_id == 1000:
        return f"/data/ymehta3/default/cfg1000{seed_letter}/training_metrics.csv"
    return f"/data/ymehta3/alexnet_pca/cfg{cfg_id}{seed_letter}/training_metrics.csv"


def load_final_accuracy(cfg_id, seed_letter):
    """Load final epoch test accuracy from training_metrics.csv."""
    path = get_csv_path(cfg_id, seed_letter)
    if not os.path.exists(path):
        print(f"  WARNING: missing {path}")
        return np.nan
    df = pd.read_csv(path)
    row = df[df["epoch"] == FINAL_EPOCH]
    if row.empty:
        print(f"  WARNING: no epoch {FINAL_EPOCH} in {path}")
        return np.nan
    return row.iloc[0]["test_acc"]


def main():
    setup_style()

    data = {}
    for cfg_id in GRAN_CFGS_ALL:
        accs = []
        for sl in SEED_LETTERS:
            acc = load_final_accuracy(cfg_id, sl)
            accs.append(acc)
        data[cfg_id] = np.array(accs)
        valid = [a for a in accs if not np.isnan(a)]
        mean = np.nanmean(accs)
        print(f"  {cfg_id:>4}-way: {mean:.2f}% ({len(valid)}/3 seeds)")

    fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))

    x_vals, y_means, y_sems, colors, markers = [], [], [], [], []

    for cfg_id in GRAN_CFGS_ALL:
        accs = data[cfg_id]
        valid = accs[~np.isnan(accs)]
        if len(valid) == 0:
            continue
        x_vals.append(cfg_id)
        y_means.append(np.mean(valid))
        y_sems.append(np.std(valid) / np.sqrt(len(valid)) if len(valid) > 1 else 0)
        colors.append(GRAN_COLORS[cfg_id])
        markers.append(GRAN_MARKERS[cfg_id])

    x_vals = np.array(x_vals)
    y_means = np.array(y_means)
    y_sems = np.array(y_sems)

    for i in range(len(x_vals) - 1):
        ax.plot(x_vals[i:i+2], y_means[i:i+2], color=colors[i],
                linewidth=1.2, zorder=1, alpha=0.5)

    for i, cfg_id in enumerate(x_vals):
        ax.errorbar(
            cfg_id, y_means[i], yerr=y_sems[i],
            fmt=markers[i], color=colors[i], markersize=8,
            markeredgecolor="white", markeredgewidth=0.8,
            ecolor=colors[i], elinewidth=1.2, capsize=3, capthick=1.0,
            zorder=3, label=f"{int(cfg_id)}-way",
        )

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(GRAN_CFGS_ALL))
    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda val, pos: str(int(val)) if int(round(val)) in set(GRAN_CFGS_ALL) else ""))
    ax.xaxis.set_minor_locator(NullLocator())

    ax.set_xlabel("Number of classes", fontsize=10, labelpad=5)
    ax.set_ylabel("Test accuracy (%)", fontsize=10, labelpad=5)
    ax.set_title("Test accuracy vs. label granularity", fontsize=11,
                 fontweight="semibold", pad=8)

    ax.yaxis.grid(True, which="major", color="#EBEBEB", linewidth=0.4, zorder=0)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.tick_params(axis="y", which="major", direction="out", length=4, width=1.0)
    ax.tick_params(axis="y", which="minor", direction="out", length=2.5, width=0.6)

    sns.despine(ax=ax, right=True, top=True, offset=4)

    plt.tight_layout()
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()
