"""
RDM heatmap comparison: Behavioral | 2-class | 1000-class.

Subsamples ~500 concepts, sorts by hierarchical clustering on the behavioral RDM
for a consistent concept ordering across all panels.

Input:  experiments/things_visualizations/data/things_viz_data.npz
Output: experiments/things_visualizations/figures/rdm_comparison.png
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

from experiments.things_visualizations.utils import load_data, FIG_DIR
from visreps.analysis.rsa import compute_rdm, compute_rdm_correlation

N_SUBSAMPLE = 500
SEED = 42


def rank_transform(rdm):
    """Rank upper triangle, mirror to lower, scale to [0, 1]."""
    n = rdm.shape[0]
    triu = np.triu_indices(n, k=1)
    ranks = rankdata(rdm[triu]) / rdm[triu].size
    ranked = np.zeros_like(rdm)
    ranked[triu] = ranks
    ranked.T[triu] = ranks
    return ranked


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    data = load_data()

    # Subsample concepts
    rng = np.random.RandomState(SEED)
    n = data["embeddings"].shape[0]
    idx = np.sort(rng.choice(n, size=N_SUBSAMPLE, replace=False))

    # Build RDMs
    rdms = {}
    for key, arr in [("Behavioral", data["embeddings"]),
                     ("2-class", data["twoclass_acts"]),
                     ("1000-class", data["thousand_acts"])]:
        rdms[key] = compute_rdm(torch.tensor(arr[idx], dtype=torch.float32)).numpy()

    # Sort by hierarchical clustering on behavioral RDM
    order = leaves_list(linkage(squareform(rdms["Behavioral"], checks=False), method="average"))
    for key in rdms:
        rdms[key] = rdms[key][np.ix_(order, order)]

    # RSA scores
    rsa_scores = {
        key: compute_rdm_correlation(torch.tensor(rdms[key]), torch.tensor(rdms["Behavioral"]),
                                     correlation="Spearman")
        for key in ["2-class", "1000-class"]
    }

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5), gridspec_kw={"wspace": 0.06})
    panels = [
        ("Behavioral", rdms["Behavioral"], None),
        ("2-class", rdms["2-class"], rsa_scores["2-class"]),
        ("1000-class", rdms["1000-class"], rsa_scores["1000-class"]),
    ]

    for ax, (title, rdm, rsa) in zip(axes, panels):
        im = ax.imshow(rank_transform(rdm), cmap="magma", vmin=0, vmax=1,
                       interpolation="nearest", aspect="equal")
        subtitle = f"$\\rho_s$ = {rsa:.3f}" if rsa is not None else "(ground truth)"
        ax.set_title(f"{title}\n{subtitle}", fontsize=12, fontweight="bold", pad=6)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.8); spine.set_color("#888888")

    axes[0].set_ylabel("Concepts (sorted by behavioral similarity)", fontsize=9, color="#555555")
    cbar = fig.colorbar(im, ax=axes.tolist(), shrink=0.72, pad=0.015, aspect=25)
    cbar.set_label("Dissimilarity (rank-normalized)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    out = os.path.join(FIG_DIR, "rdm_comparison.png")
    fig.savefig(out, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Saved: {out}")
    for key, rsa in rsa_scores.items():
        print(f"  {key} RSA: {rsa:.4f}")


if __name__ == "__main__":
    main()
