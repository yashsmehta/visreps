# Plot RSA performance comparing 1000-way classification vs PCA models (CLIP vs Scattering5 vs Scattering11)
# For early visual stream and ventral visual stream regions
# Requirements: pandas, matplotlib, numpy

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np

# --- Config ---
CSV_PATH = "logs/scat_full-vs-pcs.csv"  # Data file
DATASET = "imagenet-mini-200"  # Fixed dataset
EPOCH = 10  # Fixed epoch
REGIONS = ["early visual stream", "ventral visual stream"]  # Brain regions to analyze

# --- Load data ---
df = pd.read_csv(CSV_PATH)

# Filter for imagenet-mini-200 dataset and epoch 10
filtered = df[(df["dataset"] == DATASET) & (df["epoch"] == EPOCH)]

if filtered.empty:
    raise ValueError(f"No data found for dataset={DATASET} and epoch={EPOCH}")

# --- Process: get best layer score for each region and model type ---
# Model types:
# 1. 1000-way classification (pca_labels=False)
# 2. PCA with CLIP labels (pca_labels=True, pca_labels_folder=pca_labels_clip) - for n_classes 8, 16, 32, 64
# 3. PCA with Scattering5 labels (pca_labels=True, pca_labels_folder=pca_labels_scattering5) - for n_classes 8, 16, 32, 64
# 4. PCA with Scattering11 labels (pca_labels=True, pca_labels_folder=pca_labels_scattering11) - for n_classes 8, 16, 32, 64

results = []

for region in REGIONS:
    df_region = filtered[filtered["region"] == region]

    if df_region.empty:
        print(f"Warning: No data for region '{region}', skipping...")
        continue

    # 1. Process 1000-way classification (pca_labels=False)
    df_1k = df_region[df_region["pca_labels"] == False]
    if not df_1k.empty:
        # Average across subjects for each layer
        df_1k_avg = df_1k.groupby("layer")["score"].mean().reset_index()
        best_1k = df_1k_avg.loc[df_1k_avg["score"].idxmax()]
        best_layer = best_1k["layer"]

        # Count subjects
        n_subjects = df_1k[df_1k["layer"] == best_layer]["subject_idx"].nunique()

        results.append({
            "region": region,
            "model_type": "1k_cls",
            "pca_n_classes": None,
            "pca_labels_folder": None,
            "best_score": best_1k["score"],
            "best_layer": best_layer,
            "n_subjects": n_subjects
        })

    # 2. Process PCA with CLIP labels for each n_classes
    for n_classes in [8, 16, 32, 64]:
        df_pca_clip = df_region[(df_region["pca_labels"] == True) &
                                (df_region["pca_labels_folder"] == "pca_labels_clip") &
                                (df_region["pca_n_classes"] == n_classes)]
        if not df_pca_clip.empty:
            # Average across subjects for each layer
            df_clip_avg = df_pca_clip.groupby("layer")["score"].mean().reset_index()
            best_clip = df_clip_avg.loc[df_clip_avg["score"].idxmax()]
            best_layer = best_clip["layer"]

            # Count subjects
            n_subjects = df_pca_clip[df_pca_clip["layer"] == best_layer]["subject_idx"].nunique()

            results.append({
                "region": region,
                "model_type": "pca_clip",
                "pca_n_classes": n_classes,
                "pca_labels_folder": "pca_labels_clip",
                "best_score": best_clip["score"],
                "best_layer": best_layer,
                "n_subjects": n_subjects
            })

    # 3. Process PCA with Scattering5 labels for each n_classes
    for n_classes in [8, 16, 32, 64]:
        df_pca_scat5 = df_region[(df_region["pca_labels"] == True) &
                                 (df_region["pca_labels_folder"] == "pca_labels_scattering5") &
                                 (df_region["pca_n_classes"] == n_classes)]
        if not df_pca_scat5.empty:
            # Average across subjects for each layer
            df_scat5_avg = df_pca_scat5.groupby("layer")["score"].mean().reset_index()
            best_scat5 = df_scat5_avg.loc[df_scat5_avg["score"].idxmax()]
            best_layer = best_scat5["layer"]

            # Count subjects
            n_subjects = df_pca_scat5[df_pca_scat5["layer"] == best_layer]["subject_idx"].nunique()

            results.append({
                "region": region,
                "model_type": "pca_scat5",
                "pca_n_classes": n_classes,
                "pca_labels_folder": "pca_labels_scattering5",
                "best_score": best_scat5["score"],
                "best_layer": best_layer,
                "n_subjects": n_subjects
            })

    # 4. Process PCA with Scattering11 labels for each n_classes
    for n_classes in [8, 16, 32, 64]:
        df_pca_scat11 = df_region[(df_region["pca_labels"] == True) &
                                  (df_region["pca_labels_folder"] == "pca_labels_scattering11") &
                                  (df_region["pca_n_classes"] == n_classes)]
        if not df_pca_scat11.empty:
            # Average across subjects for each layer
            df_scat11_avg = df_pca_scat11.groupby("layer")["score"].mean().reset_index()
            best_scat11 = df_scat11_avg.loc[df_scat11_avg["score"].idxmax()]
            best_layer = best_scat11["layer"]

            # Count subjects
            n_subjects = df_pca_scat11[df_pca_scat11["layer"] == best_layer]["subject_idx"].nunique()

            results.append({
                "region": region,
                "model_type": "pca_scat11",
                "pca_n_classes": n_classes,
                "pca_labels_folder": "pca_labels_scattering11",
                "best_score": best_scat11["score"],
                "best_layer": best_layer,
                "n_subjects": n_subjects
            })

# Convert to dataframe for easier manipulation
results_df = pd.DataFrame(results)

# --- Set up x-positions and colors ---
# X-axis positions for regions (closer together)
x_base = {"early visual stream": 0, "ventral visual stream": 0.5}

# Offsets for visual separation (adjusted for 4 model types)
offset_1k = -0.11  # Leftmost position for 1000-way
# CLIP models
offset_pca_clip = {8: -0.06, 16: -0.045, 32: -0.03, 64: -0.015}
# Scattering5 models (light green)
offset_pca_scat5 = {8: 0.015, 16: 0.03, 32: 0.045, 64: 0.06}
# Scattering11 models (dark green) - rightmost
offset_pca_scat11 = {8: 0.09, 16: 0.105, 32: 0.12, 64: 0.135}

# Colors
color_1k = "#F28E2B"  # Orange for 1000-way
# Blue gradient for CLIP (light to dark)
blue_map = {8: "#A6CEE3", 16: "#6BAED6", 32: "#3182BD", 64: "#08519C"}
# Green gradient for both Scattering5 and Scattering11 (light to dark)
green_map = {8: "#A1D99B", 16: "#74C476", 32: "#31A354", 64: "#006D2C"}

# Markers
marker_1k = "s"  # Square for 1000-way
marker_clip = "^"  # Triangle up for CLIP PCA
marker_scat5 = "o"  # Circle for Scattering5 PCA
marker_scat11 = "D"  # Diamond for Scattering11 PCA

# --- Plot scatter plot ---
fig, ax = plt.subplots(figsize=(8, 6))  # Reduced height to 0.75x

# Plot 1000-way classification (orange squares)
for _, row in results_df[results_df["model_type"] == "1k_cls"].iterrows():
    x = x_base[row["region"]] + offset_1k
    y = row["best_score"]
    ax.scatter([x], [y], s=150, marker=marker_1k, c=color_1k, alpha=0.95, edgecolors='black', linewidth=0.5)

# Plot PCA CLIP points (blue triangles) and calculate means
for region in REGIONS:
    # Get all CLIP scores for this region
    clip_scores = []
    clip_x_positions = []

    for _, row in results_df[(results_df["model_type"] == "pca_clip") &
                             (results_df["region"] == region)].iterrows():
        n_classes = row["pca_n_classes"]
        if n_classes in offset_pca_clip:
            x = x_base[region] + offset_pca_clip[n_classes]
            y = row["best_score"]
            ax.scatter([x], [y], s=120, marker=marker_clip, c=blue_map[n_classes],
                      alpha=0.95, edgecolors='black', linewidth=0.5)
            clip_scores.append(y)
            clip_x_positions.append(x)

    # Draw mean line for CLIP models
    if clip_scores:
        mean_score = np.mean(clip_scores)
        x_min = min(clip_x_positions) - 0.01
        x_max = max(clip_x_positions) + 0.01
        ax.plot([x_min, x_max], [mean_score, mean_score],
               color=blue_map[32], linestyle='--', linewidth=1.5, alpha=0.7)

# Plot PCA Scattering5 points (light green triangles) and calculate means
for region in REGIONS:
    # Get all Scattering5 scores for this region
    scat5_scores = []
    scat5_x_positions = []

    for _, row in results_df[(results_df["model_type"] == "pca_scat5") &
                             (results_df["region"] == region)].iterrows():
        n_classes = row["pca_n_classes"]
        if n_classes in offset_pca_scat5:
            x = x_base[region] + offset_pca_scat5[n_classes]
            y = row["best_score"]
            ax.scatter([x], [y], s=120, marker=marker_scat5, c=green_map[n_classes],
                      alpha=0.95, edgecolors='black', linewidth=0.5)
            scat5_scores.append(y)
            scat5_x_positions.append(x)

    # Draw mean line for Scattering5 models
    if scat5_scores:
        mean_score = np.mean(scat5_scores)
        x_min = min(scat5_x_positions) - 0.01
        x_max = max(scat5_x_positions) + 0.01
        ax.plot([x_min, x_max], [mean_score, mean_score],
               color=green_map[32], linestyle='--', linewidth=1.5, alpha=0.7)

# Plot PCA Scattering11 points (dark green triangles) and calculate means
for region in REGIONS:
    # Get all Scattering11 scores for this region
    scat11_scores = []
    scat11_x_positions = []

    for _, row in results_df[(results_df["model_type"] == "pca_scat11") &
                             (results_df["region"] == region)].iterrows():
        n_classes = row["pca_n_classes"]
        if n_classes in offset_pca_scat11:
            x = x_base[region] + offset_pca_scat11[n_classes]
            y = row["best_score"]
            ax.scatter([x], [y], s=120, marker=marker_scat11, c=green_map[n_classes],
                      alpha=0.95, edgecolors='black', linewidth=0.5)
            scat11_scores.append(y)
            scat11_x_positions.append(x)

    # Draw mean line for Scattering11 models
    if scat11_scores:
        mean_score = np.mean(scat11_scores)
        x_min = min(scat11_x_positions) - 0.01
        x_max = max(scat11_x_positions) + 0.01
        ax.plot([x_min, x_max], [mean_score, mean_score],
               color=green_map[32], linestyle='--', linewidth=1.5, alpha=0.7)

# Axes formatting
ax.set_xticks([0, 0.5])
ax.set_xticklabels(REGIONS)
ax.set_xlabel("Brain Region", fontsize=12)
ax.set_ylabel("Best RSA Score", fontsize=12)
ax.set_title(f"Neural Alignment Comparison - {DATASET} (Epoch {EPOCH})", fontsize=14)
ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.6)
ax.set_xlim(-0.16, 0.68)

# Create two separate legends outside the plot
# First legend: Model types (colors and shapes)
legend1_items = [
    mlines.Line2D([], [], linestyle="None", marker=marker_1k, markersize=10,
                 color=color_1k, label="1000-way", markeredgecolor='black', markeredgewidth=0.5),
    mlines.Line2D([], [], linestyle="None", marker=marker_clip, markersize=9,
                 color=blue_map[32], label="CLIP PCA", markeredgecolor='black', markeredgewidth=0.5),
    mlines.Line2D([], [], linestyle='--', color=blue_map[32], linewidth=1.5,
                 label="CLIP mean"),
    mlines.Line2D([], [], linestyle="None", marker=marker_scat5, markersize=9,
                 color=green_map[32], label="Scat5 PCA", markeredgecolor='black', markeredgewidth=0.5),
    mlines.Line2D([], [], linestyle='--', color=green_map[32], linewidth=1.5,
                 label="Scat5 mean"),
    mlines.Line2D([], [], linestyle="None", marker=marker_scat11, markersize=9,
                 color=green_map[32], label="Scat11 PCA", markeredgecolor='black', markeredgewidth=0.5),
    mlines.Line2D([], [], linestyle='--', color=green_map[32], linewidth=1.5,
                 label="Scat11 mean"),
]

# Second legend: Number of classes (shades)
from matplotlib.patches import Patch
legend2_items = [
    Patch(facecolor=blue_map[8], label='8 classes'),
    Patch(facecolor=blue_map[16], label='16 classes'),
    Patch(facecolor=blue_map[32], label='32 classes'),
    Patch(facecolor=blue_map[64], label='64 classes'),
]

# Place legends outside the plot on the right, stacked vertically
legend1 = ax.legend(handles=legend1_items, bbox_to_anchor=(1.05, 0.95), loc='upper left',
                   frameon=True, title="Model Type")
ax.add_artist(legend1)

legend2 = ax.legend(handles=legend2_items, bbox_to_anchor=(1.05, 0.5), loc='upper left',
                   frameon=True, title="Granularity (PCA)")

plt.tight_layout()
plt.savefig("rsa_brain_region_comparison.png", dpi=200)
# plt.show()  # Comment out for non-interactive execution
print("\nPlot saved as: rsa_brain_region_comparison.png")

# --- Print summary table ---
print("\n=== Best Layer Performance Summary ===")
print(f"Dataset: {DATASET}, Epoch: {EPOCH}")
print("Averaged across 3 subjects\n")

for region in REGIONS:
    df_region = results_df[results_df["region"] == region]
    if df_region.empty:
        continue

    print(f"\n{region}:")

    # 1000-way classification
    row_1k = df_region[df_region["model_type"] == "1k_cls"]
    if not row_1k.empty:
        row = row_1k.iloc[0]
        print(f"  1000-way:        {row['best_score']:.4f} (layer: {row['best_layer']:>5})")

    # PCA CLIP models
    print("  PCA CLIP:")
    for n_classes in [8, 16, 32, 64]:
        row_clip = df_region[(df_region["model_type"] == "pca_clip") &
                            (df_region["pca_n_classes"] == n_classes)]
        if not row_clip.empty:
            row = row_clip.iloc[0]
            print(f"    {n_classes:2d} classes:   {row['best_score']:.4f} (layer: {row['best_layer']:>5})")

    # PCA Scattering5 models
    print("  PCA Scattering5:")
    for n_classes in [8, 16, 32, 64]:
        row_scat5 = df_region[(df_region["model_type"] == "pca_scat5") &
                             (df_region["pca_n_classes"] == n_classes)]
        if not row_scat5.empty:
            row = row_scat5.iloc[0]
            print(f"    {n_classes:2d} classes:   {row['best_score']:.4f} (layer: {row['best_layer']:>5})")

    # PCA Scattering11 models
    print("  PCA Scattering11:")
    for n_classes in [8, 16, 32, 64]:
        row_scat11 = df_region[(df_region["model_type"] == "pca_scat11") &
                              (df_region["pca_n_classes"] == n_classes)]
        if not row_scat11.empty:
            row = row_scat11.iloc[0]
            print(f"    {n_classes:2d} classes:   {row['best_score']:.4f} (layer: {row['best_layer']:>5})")