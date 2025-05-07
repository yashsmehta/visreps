"""
scatter_demo.py
─────────────────────────────────────────────────────────────
Synthetic demo that

1.  generates (N≈1 500, 2) embeddings in 32 Gaussian clusters,
2.  assigns a unique color per cluster (gist_ncar palette),
3.  hides axes,
4.  finds three points to illustrate:
      • p1, p2 – from *adjacent* clusters (min‑distance centroids)
      • p3     – from the cluster farthest from p1's cluster,
5.  overlays user‑supplied images on those three points.

Replace the three paths in IMG_FOR_POINT with real images
(e.g., JPG/PNG files in your working directory).
─────────────────────────────────────────────────────────────
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgba
# from sklearn.cluster import KMeans # No longer needed
import warnings
import umap
import os

# ---------------- Load Real Features -------------------
FEATURE_FILE = 'datasets/obj_cls/imagenet-mini-50/features_alexnet_pretrained_none.npz'
FEATURE_LAYER = 'fc2'
N_CLASSES = 1000
SAMPLES_PER_CLASS_ORIG = 50
SAMPLES_PER_CLASS_SUBSET = 5
SUBSET_SIZE = N_CLASSES * SAMPLES_PER_CLASS_SUBSET

print(f"Loading features from: {FEATURE_FILE} (Layer: {FEATURE_LAYER})")
if not os.path.exists(FEATURE_FILE):
    raise FileNotFoundError(f"Feature file not found: {FEATURE_FILE}")

with np.load(FEATURE_FILE) as data:
    if FEATURE_LAYER not in data:
        raise KeyError(f"Layer '{FEATURE_LAYER}' not found in {FEATURE_FILE}. Available keys: {list(data.keys())}")
    features_all = data[FEATURE_LAYER]

print(f"Original feature shape: {features_all.shape}")
if features_all.shape[0] != N_CLASSES * SAMPLES_PER_CLASS_ORIG:
    warnings.warn(f"Warning: Expected {N_CLASSES * SAMPLES_PER_CLASS_ORIG} samples, but found {features_all.shape[0]}. Subsampling assumption might be incorrect.")

# --- Subsample features (assuming ordered structure) ---
print(f"Subsampling to {SAMPLES_PER_CLASS_SUBSET} samples per class ({SUBSET_SIZE} total)...")
indices_to_keep = []
for i in range(N_CLASSES):
    start_idx = i * SAMPLES_PER_CLASS_ORIG
    indices_to_keep.extend(list(range(start_idx, start_idx + SAMPLES_PER_CLASS_SUBSET)))

features_subset = features_all[indices_to_keep]
print(f"Subsampled feature shape: {features_subset.shape}")

# --- Apply UMAP --- 
print("Applying UMAP projection...")
reducer = umap.UMAP(n_neighbors=20, min_dist=0.05, n_components=2, random_state=42)
emb = reducer.fit_transform(features_subset)
print(f"UMAP embedding shape: {emb.shape}")

# --- Shared Plotting Settings ---
scatter_kwargs = dict(s=29, linewidth=0.5, alpha=0.8, edgecolor='white')
save_kwargs = dict(format='svg', transparent=True, bbox_inches="tight")
tab20_colors = plt.cm.tab20.colors
rng = np.random.default_rng(42)

# --- Plot 1: Top/Bottom Split (Tilted Boundary & Probabilistic Noise) ---
print("Generating Tilted/Probabilistic Boundary Noise Top/Bottom split plot...")

# Rotate points slightly (e.g., 10 degrees)
angle_rad = np.deg2rad(10)
rotation_matrix = np.array([
    [np.cos(angle_rad), -np.sin(angle_rad)],
    [np.sin(angle_rad), np.cos(angle_rad)]
])
emb_rotated = emb @ rotation_matrix.T

y_median_rotated = np.median(emb_rotated[:, 1])
labels_tb_initial = (emb_rotated[:, 1] >= y_median_rotated).astype(int)

# Define a THINNER boundary threshold
std_dev_y_rotated = np.std(emb_rotated[:, 1])
# threshold_tb = 0.2 * std_dev_y_rotated # Old threshold
threshold_tb = 0.1 * std_dev_y_rotated # Reduced threshold

# Find points close to the boundary in rotated space
dist_from_boundary_tb = np.abs(emb_rotated[:, 1] - y_median_rotated)
boundary_indices_tb = np.where(dist_from_boundary_tb < threshold_tb)[0]

# Start with initial labels, then probabilistically flip boundary points
labels_tb_noisy = labels_tb_initial.copy()
for idx in boundary_indices_tb:
    # Probability of KEEPING original label increases linearly with distance from boundary
    norm_dist = dist_from_boundary_tb[idx] / threshold_tb # 0 (on boundary) to 1 (at edge)
    p_keep = norm_dist
    if rng.random() > p_keep: # More likely to flip if closer to boundary (rand > p_keep)
        labels_tb_noisy[idx] = 1 - labels_tb_initial[idx]

print(f"  Top/Bottom: Identified {len(boundary_indices_tb)} boundary points for probabilistic noise.")

colors_tb = [tab20_colors[0], tab20_colors[2]]
point_colors_tb = [colors_tb[label] for label in labels_tb_noisy]

fig_tb, ax_tb = plt.subplots(figsize=(10, 10))
ax_tb.scatter(emb[:, 0], emb[:, 1], c=point_colors_tb, **scatter_kwargs)
ax_tb.set_axis_off()
plt.savefig("scatter_umap_top_bottom_tilted_noisy.svg", **save_kwargs)
plt.close(fig_tb)
print("Saved scatter_umap_top_bottom_tilted_noisy.svg")

# --- Plot 2: Quadrant Split + Probabilistic Noise Near Boundaries ---
print("Generating Quadrant split plot with probabilistic boundary noise...")
center_x = np.median(emb[:, 0])
center_y = np.median(emb[:, 1])

# Initial quadrant assignment based on ORIGINAL embeddings
labels_quad_initial = np.zeros(emb.shape[0], dtype=int)
labels_quad_initial[(emb[:, 0] < center_x) & (emb[:, 1] < center_y)] = 0 # BL
labels_quad_initial[(emb[:, 0] >= center_x) & (emb[:, 1] < center_y)] = 1 # BR
labels_quad_initial[(emb[:, 0] < center_x) & (emb[:, 1] >= center_y)] = 2 # TL
labels_quad_initial[(emb[:, 0] >= center_x) & (emb[:, 1] >= center_y)] = 3 # TR

# Define THINNER boundary thresholds
# threshold_x = 0.15 * np.std(emb[:, 0]) # Old threshold
# threshold_y = 0.15 * np.std(emb[:, 1]) # Old threshold
threshold_x = 0.07 * np.std(emb[:, 0]) # Reduced threshold
threshold_y = 0.07 * np.std(emb[:, 1]) # Reduced threshold

# Find points close to EITHER the vertical or horizontal boundary
dist_x = np.abs(emb[:, 0] - center_x)
dist_y = np.abs(emb[:, 1] - center_y)
# Find indices near vertical boundary (but not too close to center horizontally)
near_vertical_idx = np.where((dist_x < threshold_x) & (dist_y >= threshold_y))[0]
# Find indices near horizontal boundary (but not too close to center vertically)
near_horizontal_idx = np.where((dist_y < threshold_y) & (dist_x >= threshold_x))[0]
# Optional: Handle points near the center (dist_x < th_x AND dist_y < th_y) separately if needed
# Currently they will keep their original label as they aren't in near_vertical/horizontal

# Start with initial labels, then apply boundary-specific probabilistic noise
labels_quad_noisy = labels_quad_initial.copy()

# Process points near vertical boundary
for idx in near_vertical_idx:
    norm_dist = dist_x[idx] / threshold_x
    p_keep = norm_dist
    if rng.random() > p_keep:
        original_label = labels_quad_initial[idx]
        # Flip horizontally (BL <-> BR, TL <-> TR)
        if original_label == 0: alt_label = 1
        elif original_label == 1: alt_label = 0
        elif original_label == 2: alt_label = 3
        else: alt_label = 2 # original_label == 3
        labels_quad_noisy[idx] = alt_label

# Process points near horizontal boundary
for idx in near_horizontal_idx:
    norm_dist = dist_y[idx] / threshold_y
    p_keep = norm_dist
    if rng.random() > p_keep:
        original_label = labels_quad_initial[idx]
        # Flip vertically (BL <-> TL, BR <-> TR)
        if original_label == 0: alt_label = 2
        elif original_label == 1: alt_label = 3
        elif original_label == 2: alt_label = 0
        else: alt_label = 1 # original_label == 3
        labels_quad_noisy[idx] = alt_label

print(f"  Quadrants: Processed {len(near_vertical_idx)} near vertical, {len(near_horizontal_idx)} near horizontal boundaries for noise.")

colors_quad = [tab20_colors[0], tab20_colors[2], tab20_colors[4], tab20_colors[6]]
point_colors_quad = [colors_quad[label] for label in labels_quad_noisy]

fig_quad, ax_quad = plt.subplots(figsize=(10, 10))
ax_quad.scatter(emb[:, 0], emb[:, 1], c=point_colors_quad, **scatter_kwargs)
ax_quad.set_axis_off()
# plt.savefig("scatter_umap_quadrants_boundary_noise.svg", **save_kwargs) # Old name
plt.savefig("scatter_umap_quadrants_prob_noise.svg", **save_kwargs) # New name
plt.close(fig_quad)
# print("Saved scatter_umap_quadrants_boundary_noise.svg")
print("Saved scatter_umap_quadrants_prob_noise.svg")

print("Done.")