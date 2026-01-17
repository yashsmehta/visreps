"""Visualize Binary PC RSA Results."""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load binary PC data
df = pd.read_csv("experiments/binary_pc_rsa/binary_pc_rsa.csv")

# Load AlexNet baseline
df_cnn = pd.read_csv("logs/binary_pc_exp_CNN.csv")
df_cnn = df_cnn.rename(columns={"compare_rsm_correlation": "correlation"})

# Filter CNN to only subjects present in binary PC data (for fair comparison)
common_subjects = df["subject_idx"].unique()
df_cnn = df_cnn[df_cnn["subject_idx"].isin(common_subjects)]

# Layer mapping per region
layer_for_region = {
    "early visual stream": "conv4",
    "ventral visual stream": "fc1",
}

# Build baseline dict: (region, correlation, epoch) -> score
baseline = {}
for region, layer in layer_for_region.items():
    region_layer_df = df_cnn[(df_cnn["layer"] == layer) & (df_cnn["region"] == region)]
    for epoch in [0, 20]:
        epoch_df = region_layer_df[region_layer_df["epoch"] == epoch]
        for corr in ["Spearman", "Kendall"]:
            corr_df = epoch_df[epoch_df["correlation"] == corr]
            if len(corr_df) > 0:
                baseline[(region, corr, epoch)] = corr_df["score"].mean()

# Average across subjects
df_avg = df.groupby(["n_pcs", "region", "weighted", "correlation"])["score"].mean().reset_index()

# Setup: Create two separate figures, one per region
# Each figure has 1 row x 2 cols (Spearman | Kendall)
regions = ["early visual stream", "ventral visual stream"]
correlations = ["Spearman", "Kendall"]
colors = {"Weighted": "#E63946", "Non-weighted": "#F4A261"}  # Red, Orange

for region in regions:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    layer = layer_for_region[region]

    for col, corr in enumerate(correlations):
        ax = axes[col]
        subset = df_avg[(df_avg["correlation"] == corr) & (df_avg["region"] == region)]

        # Add baselines FIRST (so they're behind)
        # Epoch 0 (untrained) - gray dashed
        score_untrained = baseline.get((region, corr, 0), None)
        if score_untrained is not None:
            ax.axhline(score_untrained, color="#808080", linestyle="--", linewidth=2,
                       label=f"AlexNet {layer} (untrained)", zorder=1)

        # Epoch 20 (trained) - blue dashed
        score_trained = baseline.get((region, corr, 20), None)
        if score_trained is not None:
            ax.axhline(score_trained, color="#457B9D", linestyle="--", linewidth=2,
                       label=f"AlexNet {layer} (trained)", zorder=1)

        for weighted, label in [(True, "Weighted"), (False, "Non-weighted")]:
            data = subset[subset["weighted"] == weighted].sort_values("n_pcs")
            ax.plot(data["n_pcs"], data["score"], marker="o", markersize=4, linewidth=2,
                    color=colors[label], label=label, zorder=2)

        ax.set_title(f"{corr}", fontsize=11, fontweight='bold')
        ax.set_ylabel("RSA Score" if col == 0 else "")
        ax.set_xlabel("Number of PCs")
        ax.grid(True, alpha=0.3)

        # X-axis: major ticks at 5, 10, 15, 20; minor ticks at every integer
        ax.set_xticks([5, 10, 15, 20])
        ax.set_xticks(np.arange(2, 21, 1), minor=True)
        ax.set_xlim(1.5, 20.5)

    # Create filename based on region
    region_short = region.replace(" visual stream", "").replace(" ", "_")
    filename = f"experiments/binary_pc_rsa/binary_pc_rsa_{region_short}.png"

    plt.suptitle(f"Binary PC RSA: {region.title()} (averaged across subjects)", fontsize=13, fontweight='bold')

    # Add shared legend outside the subplots (to the right)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5),
               fontsize=9, frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved to {filename}")
