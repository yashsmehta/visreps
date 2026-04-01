"""
Supplementary Figure S12: Internal representation analysis across granularity.

2x2 figure:
  (a) FC1 Eigenspectrum (log-log) — one line per granularity
  (b) Participation ratio across layers
  (c) Two-NN intrinsic dimension across layers
  (d) Hoyer sparsity across layers

One line per granularity level using GRAN_COLORS from fig_utils.
Mean +/- SEM across 3 seeds.

Usage (from project root):
    python manuscript/figures/supplementary/supp_s12_representation_summary.py
"""

import json
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, ".")
from manuscript.figures.fig_utils import setup_style, GRAN_COLORS, GRAN_MARKERS, COARSE_CFGS

setup_style()

# ── Configuration ─────────────────────────────────────────────────────
DATA_PATH = "experiments/representation_analysis/figs/representation_summary_data.json"
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "supp_s12_representation_summary.png")

EIGENSPECTRUM_LAYER = "fc1"
N_EIGEN_COMPONENTS = 100

ALL_IDS = ["untrained"] + COARSE_CFGS + [1000]

LAYER_LABELS = {
    "conv1": "Conv1", "conv2": "Conv2", "conv3": "Conv3",
    "conv4": "Conv4", "conv5": "Conv5", "fc1": "FC1", "fc2": "FC2",
}

# Styling
UNTRAINED_STYLE = {"color": "#AAAAAA", "linestyle": "--", "linewidth": 1.4, "markersize": 4, "zorder": 1}
FINE_STYLE = {"color": GRAN_COLORS[1000], "linestyle": "-", "linewidth": 1.8, "markersize": 5, "zorder": 10}


def get_style(cfg_id):
    if cfg_id == "untrained":
        return UNTRAINED_STYLE
    elif cfg_id == 1000:
        return FINE_STYLE
    else:
        return {"color": GRAN_COLORS[cfg_id], "linestyle": "-", "linewidth": 1.4,
                "markersize": 4.5, "zorder": 5}


def get_label(cfg_id):
    if cfg_id == "untrained":
        return "Untrained"
    return f"{cfg_id}-way"


def get_marker(cfg_id):
    if cfg_id == "untrained":
        return None
    return GRAN_MARKERS.get(cfg_id, "o")


def load_data():
    """Load precomputed representation summary data from JSON."""
    with open(DATA_PATH) as f:
        data = json.load(f)
    layers = data["layers"]
    all_results = {}
    for key, seed_list in data["results"].items():
        cfg_id = key if key == "untrained" else int(key)
        all_results[cfg_id] = []
        for entry in seed_list:
            metrics = {}
            for metric_name, layer_dict in entry.items():
                metrics[metric_name] = {}
                for layer, val in layer_dict.items():
                    if isinstance(val, list):
                        metrics[metric_name][layer] = np.array(val)
                    else:
                        metrics[metric_name][layer] = val
            all_results[cfg_id].append(metrics)
    return all_results, layers


def aggregate_seeds(seed_results, layers, metric_key):
    n_seeds = len(seed_results)
    vals = np.array([[r[metric_key][l] for l in layers] for r in seed_results])
    means = vals.mean(axis=0)
    sems = vals.std(axis=0) / np.sqrt(n_seeds) if n_seeds > 1 else np.zeros_like(means)
    return means, sems


def main():
    all_results, layers = load_data()

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # ── (a) Eigenspectrum ─────────────────────────────────────────────
    ax = axes[0, 0]
    layer = EIGENSPECTRUM_LAYER
    for cfg_id in ALL_IDS:
        if cfg_id not in all_results:
            continue
        style = get_style(cfg_id)
        label = get_label(cfg_id)
        spectra = []
        for r in all_results[cfg_id]:
            eigs = r["eigenvalues"][layer]
            n_plot = min(N_EIGEN_COMPONENTS, len(eigs))
            spectra.append(eigs[:n_plot] / eigs.sum())
        spectra = np.array(spectra)
        mean_spec = spectra.mean(axis=0)
        comps = np.arange(1, len(mean_spec) + 1)
        if len(spectra) > 1:
            sem = spectra.std(axis=0) / np.sqrt(len(spectra))
            ax.fill_between(comps, mean_spec - sem, mean_spec + sem,
                            color=style["color"], alpha=0.08, linewidth=0)
        ax.plot(comps, mean_spec, color=style["color"],
                linestyle=style["linestyle"], linewidth=style["linewidth"] + 0.2,
                label=label, zorder=style.get("zorder", 5))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Component", fontsize=9)
    ax.set_ylabel("Fraction of variance", fontsize=9)
    ax.text(-0.08, 1.08, "a", transform=ax.transAxes, fontsize=13, fontweight="bold", va="top", ha="left")
    ax.set_title("FC1 eigenspectrum", loc="left", fontsize=10, fontweight="bold", pad=8)
    sns.despine(ax=ax, offset=5)

    # ── Panels (b)-(d): layer metrics ─────────────────────────────────
    metric_configs = [
        (axes[0, 1], "pr", "Participation ratio", "Effective dimensionality", True, "b"),
        (axes[1, 0], "twonn", "Intrinsic dimension", "Two-NN intrinsic dimension", False, "c"),
        (axes[1, 1], "sparsity", "Hoyer sparsity", "Activation sparsity", False, "d"),
    ]

    for ax, metric_key, ylabel, title, log_y, panel_label in metric_configs:
        x = np.arange(len(layers))
        for cfg_id in ALL_IDS:
            if cfg_id not in all_results:
                continue
            style = get_style(cfg_id)
            label = get_label(cfg_id)
            marker = get_marker(cfg_id)
            means, sems = aggregate_seeds(all_results[cfg_id], layers, metric_key)
            n_seeds = len(all_results[cfg_id])
            if n_seeds > 1:
                ax.fill_between(x, means - sems, means + sems,
                                color=style["color"], alpha=0.10, linewidth=0)
            plot_kw = dict(color=style["color"], linestyle=style["linestyle"],
                           linewidth=style["linewidth"],
                           markersize=style["markersize"],
                           markeredgecolor="white", markeredgewidth=0.6,
                           label=label, zorder=style.get("zorder", 5))
            if marker:
                plot_kw["marker"] = marker
            ax.plot(x, means, **plot_kw)

        ax.set_xticks(x)
        ax.set_xticklabels([LAYER_LABELS.get(l, l) for l in layers], fontsize=7.5)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlabel("Layer", fontsize=9)
        ax.text(-0.08, 1.08, panel_label, transform=ax.transAxes, fontsize=13, fontweight="bold", va="top", ha="left")
        ax.set_title(title, loc="left", fontsize=10, fontweight="bold", pad=8)
        if log_y:
            ax.set_yscale("log")
        if metric_key == "sparsity":
            ax.set_ylim(0.1, 1.02)
        ax.yaxis.grid(True, which="major", color="#EBEBEB", linewidth=0.4, zorder=0)
        sns.despine(ax=ax, offset=5)

    # ── Legend ─────────────────────────────────────────────────────────
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, -0.005), frameon=False,
               fontsize=7.5, ncol=4, columnspacing=1.5,
               handlelength=2.2, handletextpad=0.4, labelspacing=0.4)

    plt.tight_layout(h_pad=2.5, w_pad=2.5)
    fig.subplots_adjust(bottom=0.09)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
