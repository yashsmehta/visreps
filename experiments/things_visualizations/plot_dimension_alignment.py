"""
Per-dimension alignment: which of the 66 behavioral dimensions are better
captured by 2-class vs 1000-class models?

For each dimension d, computes Spearman rho between |emb_i[d] - emb_j[d]|
(behavioral) and the model's Pearson RDM upper triangle.

Input:  experiments/things_visualizations/data/things_viz_data.npz
Output: experiments/things_visualizations/figures/dimension_alignment.png
        experiments/things_visualizations/figures/dimension_difference.png
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from experiments.things_visualizations.utils import load_data, FIG_DIR
from visreps.analysis.rsa import compute_rdm

TOP_N = 25
COLOR_2C, COLOR_1K = "#2166ac", "#b2182b"


def compute_per_dimension_rsa(model_acts, embeddings):
    """Spearman rho between per-dimension behavioral dissimilarity and model RDM."""
    n, n_dims = model_acts.shape[0], embeddings.shape[1]
    model_rdm = compute_rdm(torch.tensor(model_acts, dtype=torch.float32)).numpy()
    triu = np.triu_indices(n, k=1)
    model_dissim = model_rdm[triu]

    scores = np.zeros(n_dims)
    for d in range(n_dims):
        rho, _ = spearmanr(model_dissim, np.abs(embeddings[triu[0], d] - embeddings[triu[1], d]))
        scores[d] = rho if not np.isnan(rho) else 0.0
    return scores


def save_fig(fig, path):
    plt.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {path}")


def plot_grouped_bars(scores_two, scores_thou, labels, output_path):
    """Horizontal grouped bar chart for top dimensions by |difference|."""
    diff = scores_two - scores_thou
    order = np.argsort(np.abs(diff))[::-1][:TOP_N]
    order = order[np.argsort(diff[order])]

    fig, ax = plt.subplots(figsize=(8, 0.42 * TOP_N))
    y = np.arange(TOP_N)
    h = 0.35
    ax.barh(y + h/2, scores_two[order], h, color=COLOR_2C, label="2-class", zorder=3)
    ax.barh(y - h/2, scores_thou[order], h, color=COLOR_1K, label="1000-class", zorder=3)
    ax.set_yticks(y, [labels[i] for i in order], fontsize=8)
    ax.set_xlabel("Spearman $\\rho$ (dimension alignment)", fontsize=10)
    ax.set_title(f"Top {TOP_N} dimensions by alignment difference", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.axvline(0, color="gray", lw=0.5, zorder=1)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    save_fig(fig, output_path)


def plot_difference(scores_two, scores_thou, labels, output_path):
    """Bar chart of (2-class - 1000-class) for all dimensions, sorted."""
    diff = scores_two - scores_thou
    order = np.argsort(diff)[::-1]

    fig, ax = plt.subplots(figsize=(8, 0.34 * len(labels)))
    colors = [COLOR_2C if d > 0 else COLOR_1K for d in diff[order]]
    ax.barh(np.arange(len(labels)), diff[order], color=colors, edgecolor="none", zorder=3)
    ax.set_yticks(np.arange(len(labels)), [labels[i] for i in order], fontsize=7)
    ax.set_xlabel("$\\Delta$ Spearman $\\rho$ (2-class $-$ 1000-class)", fontsize=10)
    ax.set_title("Per-dimension alignment difference", fontsize=12, fontweight="bold")
    ax.axvline(0, color="black", lw=0.8, zorder=4)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.text(0.98, 0.02, "blue = 2-class better", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=8, color=COLOR_2C, fontstyle="italic")
    ax.text(0.98, 0.06, "red = 1000-class better", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=8, color=COLOR_1K, fontstyle="italic")
    save_fig(fig, output_path)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    data = load_data()
    labels = [str(l) for l in data["dimension_labels"]]

    print(f"Computing per-dimension RSA ({len(labels)} dimensions)...")
    scores_two = compute_per_dimension_rsa(data["twoclass_acts"], data["embeddings"])
    scores_thou = compute_per_dimension_rsa(data["thousand_acts"], data["embeddings"])

    diff = scores_two - scores_thou
    print(f"  2-class better on {(diff > 0).sum()}/{len(labels)} dimensions")
    print(f"  Mean diff: {diff.mean():.4f} (positive = 2-class better)")

    plot_grouped_bars(scores_two, scores_thou, labels, os.path.join(FIG_DIR, "dimension_alignment.png"))
    plot_difference(scores_two, scores_thou, labels, os.path.join(FIG_DIR, "dimension_difference.png"))


if __name__ == "__main__":
    main()
