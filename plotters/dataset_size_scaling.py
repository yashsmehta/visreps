# Plot RSA performance as a function of dataset size
# Compares 1k CLS vs PCA models (8/16/32/64 classes) at their best layers
# Requirements: pandas, matplotlib, scipy

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
from scipy import stats

# --- Config ---
CSV_PATH_MINI = "logs/imagenet_mini_full-vs-pcs.csv"  # Mini datasets
CSV_PATH_FULL = "logs/imagenet_full-vs-pcs.csv"       # Full ImageNet
REGION   = "early visual stream"

# Dataset configurations with their convergence epochs
DATASETS = {
    "imagenet-mini-10": {"display": "10K", "epoch": 100},
    "imagenet-mini-50": {"display": "50K", "epoch": 50},
    "imagenet-mini-200": {"display": "200K", "epoch": 25},
    "imagenet": {"display": "1.2M", "epoch": 20}
}

# --- Load data ---
# Load both CSV files
df_mini = pd.read_csv(CSV_PATH_MINI)
df_full = pd.read_csv(CSV_PATH_FULL)

# Filter by region and combine with appropriate epochs
filtered_list = []
for dataset, config in DATASETS.items():
    if dataset == "imagenet":
        df_temp = df_full[(df_full["region"] == REGION) &
                         (df_full["epoch"] == config["epoch"]) &
                         (df_full["dataset"] == dataset)].copy()
    else:
        df_temp = df_mini[(df_mini["region"] == REGION) &
                         (df_mini["epoch"] == config["epoch"]) &
                         (df_mini["dataset"] == dataset)].copy()
    filtered_list.append(df_temp)

filtered = pd.concat(filtered_list, ignore_index=True)

if filtered.empty:
    raise ValueError("No rows after filtering. Check REGION/EPOCH or CSV_PATH.")

# --- Process: get best layer score for each dataset/model config ---
# First average across subjects, then find best layer
# Also collect individual subject scores for significance testing
results = []
subject_scores = {}  # Store individual subject scores for significance testing

for dataset, config in DATASETS.items():
    display_size = config["display"]
    df_dataset = filtered[filtered["dataset"] == dataset]

    if df_dataset.empty:
        print(f"Warning: No data for {dataset} at epoch {config['epoch']}, skipping...")
        continue

    # For 1k CLS (pca_labels=False): average across subjects, then find best layer
    df_1k = df_dataset[df_dataset["pca_labels"] == False]
    if not df_1k.empty:
        # Check subject counts
        subject_counts = df_1k.groupby("layer")["subject_idx"].nunique()
        incomplete = subject_counts[subject_counts < 4]
        if not incomplete.empty:
            print(f"Warning: {dataset} 1k CLS has incomplete subjects:")
            for layer, count in incomplete.items():
                print(f"  Layer {layer}: {count} subjects")

        # Average across subjects for each layer
        df_1k_avg = df_1k.groupby("layer")["score"].mean().reset_index()
        best_1k = df_1k_avg.loc[df_1k_avg["score"].idxmax()]
        best_layer = best_1k["layer"]

        # Store individual subject scores for the best layer
        best_layer_scores = df_1k[df_1k["layer"] == best_layer].sort_values("subject_idx")["score"].values
        subject_scores[(dataset, "1k_cls", None)] = best_layer_scores

        # Count subjects used in averaging for the best layer
        n_subjects = df_1k[df_1k["layer"] == best_layer]["subject_idx"].nunique()

        results.append({
            "dataset": dataset,
            "display_size": display_size,
            "model_type": "1k_cls",
            "pca_n_classes": None,
            "best_score": best_1k["score"],
            "best_layer": best_layer,
            "n_subjects": n_subjects,
            "epoch": config["epoch"]
        })

    # For PCA models (pca_labels=True): average across subjects, then find best layer per n_classes
    df_pca = df_dataset[df_dataset["pca_labels"] == True]
    for n_classes in [8, 16, 32, 64]:
        df_subset = df_pca[df_pca["pca_n_classes"] == n_classes]
        if not df_subset.empty:
            # Check subject counts
            subject_counts = df_subset.groupby("layer")["subject_idx"].nunique()
            incomplete = subject_counts[subject_counts < 4]
            if not incomplete.empty:
                print(f"Warning: {dataset} PCA-{n_classes} has incomplete subjects:")
                for layer, count in incomplete.items():
                    print(f"  Layer {layer}: {count} subjects")

            # Average across subjects for each layer
            df_subset_avg = df_subset.groupby("layer")["score"].mean().reset_index()
            best = df_subset_avg.loc[df_subset_avg["score"].idxmax()]
            best_layer = best["layer"]

            # Store individual subject scores for the best layer
            best_layer_scores = df_subset[df_subset["layer"] == best_layer].sort_values("subject_idx")["score"].values
            subject_scores[(dataset, "pca", n_classes)] = best_layer_scores

            # Count subjects used in averaging for the best layer
            n_subjects = df_subset[df_subset["layer"] == best_layer]["subject_idx"].nunique()

            results.append({
                "dataset": dataset,
                "display_size": display_size,
                "model_type": "pca",
                "pca_n_classes": n_classes,
                "best_score": best["score"],
                "best_layer": best_layer,
                "n_subjects": n_subjects,
                "epoch": config["epoch"]
            })

# Convert to dataframe for easier manipulation
results_df = pd.DataFrame(results)

# --- Perform significance testing ---
# Compare 1k CLS with each PCA model for each dataset
significance_results = {}
alpha = 0.05  # Significance level

for dataset in DATASETS.keys():
    # Get 1k CLS scores
    key_1k = (dataset, "1k_cls", None)
    if key_1k not in subject_scores:
        continue
    scores_1k = subject_scores[key_1k]

    # Compare with each PCA model
    for n_classes in [8, 16, 32, 64]:
        key_pca = (dataset, "pca", n_classes)
        if key_pca in subject_scores:
            scores_pca = subject_scores[key_pca]

            # Paired t-test (same subjects/stimuli)
            if len(scores_1k) == len(scores_pca) and len(scores_1k) > 0:
                t_stat, p_value = stats.ttest_rel(scores_1k, scores_pca)
                is_significant = p_value < alpha
                significance_results[(dataset, n_classes)] = is_significant

                # Print significance results
                if is_significant:
                    print(f"{dataset} - PCA-{n_classes} vs 1k CLS: p={p_value:.4f} *")
            else:
                print(f"Warning: Cannot perform t-test for {dataset} PCA-{n_classes} - mismatched sample sizes")

# --- Set up x-positions ---
# Order datasets by size
size_order = ["10K", "50K", "200K", "1.2M"]
x_base = {size: i for i, size in enumerate(size_order)}

# Offsets for visual separation
offset_1k = -0.15
# Cluster PCA models on the right
offset_pca_cluster = {8: 0.08, 16: 0.09, 32: 0.10, 64: 0.11}

# Colors & markers (same as original)
color_1k = "#F28E2B"  # orange
blue_map = {8: "#A6CEE3", 16: "#6BAED6", 32: "#3182BD", 64: "#08519C"}
marker_1k = "s"  # square
marker_pca = "^"  # triangle

# --- Plot ---
plt.figure(figsize=(10, 6))

# Plot 1k CLS points (orange squares)
for _, row in results_df[results_df["model_type"] == "1k_cls"].iterrows():
    x = x_base[row["display_size"]] + offset_1k
    y = row["best_score"]
    plt.scatter([x], [y], s=120, marker=marker_1k, c=color_1k, alpha=0.95)

# Plot PCA points (blue triangles)
for _, row in results_df[results_df["model_type"] == "pca"].iterrows():
    n_classes = int(row["pca_n_classes"])
    x = x_base[row["display_size"]] + offset_pca_cluster[n_classes]
    y = row["best_score"]
    plt.scatter([x], [y], s=120, marker=marker_pca, c=blue_map[n_classes], alpha=0.95)

    # Add star if significantly different from 1k CLS
    if (row["dataset"], n_classes) in significance_results and significance_results[(row["dataset"], n_classes)]:
        plt.text(x + 0.012, y + 0.0005, "*", fontsize=12, fontweight="bold", color="gray", ha="left", va="bottom")

# Connect points across dataset sizes for each model type (optional)
connect_lines = True
if connect_lines:
    # Line for 1k CLS
    df_1k_sorted = results_df[results_df["model_type"] == "1k_cls"].sort_values("display_size",
                                                                                 key=lambda x: x.map({s: i for i, s in enumerate(size_order)}))
    if len(df_1k_sorted) > 1:
        xs = [x_base[row["display_size"]] + offset_1k for _, row in df_1k_sorted.iterrows()]
        ys = df_1k_sorted["best_score"].tolist()
        plt.plot(xs, ys, color=color_1k, alpha=0.3, linewidth=1, linestyle='--')

    # Lines for each PCA n_classes
    for n_classes in [8, 16, 32, 64]:
        df_pca_n = results_df[(results_df["model_type"] == "pca") &
                              (results_df["pca_n_classes"] == n_classes)].sort_values("display_size",
                                                                                      key=lambda x: x.map({s: i for i, s in enumerate(size_order)}))
        if len(df_pca_n) > 1:
            xs = [x_base[row["display_size"]] + offset_pca_cluster[n_classes] for _, row in df_pca_n.iterrows()]
            ys = df_pca_n["best_score"].tolist()
            plt.plot(xs, ys, color=blue_map[n_classes], alpha=0.3, linewidth=1, linestyle='--')

# Axes & cosmetics
plt.xticks(range(len(size_order)), size_order)
plt.xlabel("Dataset Size (# Images)", fontsize=12)
plt.ylabel("Best RSA Score", fontsize=12)
plt.title(f"{REGION} — Best Layer Performance by Dataset Size (at convergence)", fontsize=14)
plt.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
plt.margins(x=0.1)

# Legend
legend_items = [
    mlines.Line2D([], [], linestyle="None", marker=marker_1k, markersize=9, color=color_1k, label="1k CLS"),
    mlines.Line2D([], [], linestyle="None", marker=marker_pca, markersize=9, color=blue_map[8], label="8 CLS"),
    mlines.Line2D([], [], linestyle="None", marker=marker_pca, markersize=9, color=blue_map[16], label="16 CLS"),
    mlines.Line2D([], [], linestyle="None", marker=marker_pca, markersize=9, color=blue_map[32], label="32 CLS"),
    mlines.Line2D([], [], linestyle="None", marker=marker_pca, markersize=9, color=blue_map[64], label="64 CLS"),
]
# Add star explanation to legend
legend_items.append(mlines.Line2D([], [], linestyle="None", marker="$*$", markersize=10, color="gray", label="p < 0.05"))
plt.legend(handles=legend_items, loc="best", frameon=True)

plt.tight_layout()
plt.savefig("rsa_dataset_size_scaling.png", dpi=200)
# plt.show()  # Comment out for non-interactive execution
print("\nPlot saved as: rsa_dataset_size_scaling.png")

# --- Print summary table ---
print("\n=== Significance Testing Results ===")
print(f"Paired t-test comparing 1k CLS with PCA models (alpha = {alpha})")
print("\n=== Best Layer Performance Summary ===")
print(f"Region: {REGION}\n")
print("Convergence epochs used:")
for dataset, config in DATASETS.items():
    print(f"  {config['display']:>6} ({dataset}): epoch {config['epoch']}")
print()

for dataset, config in DATASETS.items():
    df_subset = results_df[results_df["dataset"] == dataset]
    if df_subset.empty:
        continue

    print(f"\n{config['display']} ({dataset}, epoch {config['epoch']}):")

    # 1k CLS
    row_1k = df_subset[df_subset["model_type"] == "1k_cls"]
    if not row_1k.empty:
        row = row_1k.iloc[0]
        print(f"  1k CLS:  {row['best_score']:.4f} (layer: {row['best_layer']}) — averaging across {int(row.get('n_subjects', np.nan))} subjects")

    # PCA models
    for n_classes in [8, 16, 32, 64]:
        row_pca = df_subset[(df_subset["model_type"] == "pca") &
                           (df_subset["pca_n_classes"] == n_classes)]
        if not row_pca.empty:
            row = row_pca.iloc[0]
            print(f"  {n_classes:2d} CLS:  {row['best_score']:.4f} (layer: {row['best_layer']}) — averaging across {int(row.get('n_subjects', np.nan))} subjects")