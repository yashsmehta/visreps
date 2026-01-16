"""Visualize Binary PC RSA Results."""

import pandas as pd
import matplotlib.pyplot as plt

# Load binary PC data
df = pd.read_csv("experiments/results/binary_pc_rsa.csv")

# Load AlexNet baseline (fc2 layer)
df_cnn = pd.read_csv("logs/binary_pc_exp_CNN.csv")
df_cnn = df_cnn[df_cnn["layer"] == "fc2"]  # fc2 only

# Map column names for consistency
df_cnn = df_cnn.rename(columns={"compare_rsm_correlation": "correlation"})

# Average AlexNet fc2 across subjects for each (region, correlation)
baseline = df_cnn.groupby(["region", "correlation"])["score"].mean().to_dict()

# Average binary PC across subjects
df_avg = df.groupby(["n_pcs", "region", "weighted", "correlation"])["score"].mean().reset_index()

# Setup 2x2 grid: rows=correlation (Spearman/Kendall), cols=region (early/ventral)
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

regions = ["early visual stream", "ventral visual stream"]
correlations = ["Spearman", "Kendall"]
colors = {"Weighted": "#E63946", "Non-weighted": "#457B9D"}

for row, corr in enumerate(correlations):
    for col, region in enumerate(regions):
        ax = axes[row, col]
        subset = df_avg[(df_avg["correlation"] == corr) & (df_avg["region"] == region)]
        
        for weighted, label in [(True, "Weighted"), (False, "Non-weighted")]:
            data = subset[subset["weighted"] == weighted].sort_values("n_pcs")
            ax.plot(data["n_pcs"], data["score"], 
                    marker="o", markersize=4, linewidth=2,
                    color=colors[label], label=label)
        
        # Add AlexNet fc2 baseline
        baseline_score = baseline.get((region, corr), None)
        if baseline_score is not None:
            ax.axhline(baseline_score, color="#2A9D8F", linestyle="--", linewidth=2, label="AlexNet fc2")
        
        ax.set_title(f"{corr} | {region.replace(' visual stream', '').title()}", fontsize=11, fontweight='bold')
        ax.set_ylabel("RSA Score" if col == 0 else "")
        ax.set_xlabel("Number of PCs" if row == 1 else "")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)

plt.suptitle("Binary PC RSA: Neural Alignment (averaged across subjects)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("experiments/results/binary_pc_rsa_plot.png", dpi=150, bbox_inches='tight')
plt.show()
print("Saved to experiments/results/binary_pc_rsa_plot.png")
