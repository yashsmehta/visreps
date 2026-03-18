"""Find triplet candidates for Figure 3 image insets.

Looks for triplets of concepts from 3 different super-categories where:
- Behavioral & CLIP-8 distances are large (well separated)
- AlexNet & ViT distances are moderate (close but NOT overlapping)

Usage (from project root):
    python manuscript/figures/fig3/find_triplet_candidates.py
"""

import os
import sys
import itertools

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

sys.path.insert(0, "manuscript/figures/fig3")
from plot_pc_scatter import (
    load_super_categories, SUPER_ORDER, SUPER_COLORS, l2_normalize, compute_pca,
)

# ── Load data ────────────────────────────────────────────────────────────

print("Loading data...")
behav_data = np.load("experiments/things_visualizations/data/things_viz_data.npz", allow_pickle=True)
embeddings = behav_data["embeddings"]  # (1854, 66)
concept_names = behav_data["concept_names"]  # (1854,)

act_data = np.load("manuscript/figures/fig4/activations.npz", allow_pickle=True)
clip8_fc1 = l2_normalize(act_data["clip8_fc1"])  # (1854, 4096)

alexnet_data = np.load("manuscript/figures/fig4/pretrained_alexnet_fc1.npz", allow_pickle=True)
alexnet_fc1 = l2_normalize(alexnet_data["fc1"])  # (1854, 4096)

vit_data = np.load("manuscript/figures/fig3/pretrained_vit_things.npz", allow_pickle=True)
vit_block5 = l2_normalize(vit_data["block5"])  # (1854, 151296)

n_concepts = len(concept_names)
print(f"Loaded {n_concepts} concepts")

# ── Super-categories ──────────────────────────────────────────────────────
labels = load_super_categories(n_concepts)

# ── PCA (2 components) for each representation ──────────────────────────
print("Computing PCA...")
behav_pca, behav_var = compute_pca(embeddings)
clip8_pca, clip8_var = compute_pca(clip8_fc1)
alexnet_pca, alexnet_var = compute_pca(alexnet_fc1)
vit_pca, vit_var = compute_pca(vit_block5)

print(f"Variance explained:")
print(f"  Behavioral: PC1={behav_var[0]:.1f}%, PC2={behav_var[1]:.1f}%")
print(f"  CLIP-8:     PC1={clip8_var[0]:.1f}%, PC2={clip8_var[1]:.1f}%")
print(f"  AlexNet:    PC1={alexnet_var[0]:.1f}%, PC2={alexnet_var[1]:.1f}%")
print(f"  ViT:        PC1={vit_var[0]:.1f}%, PC2={vit_var[1]:.1f}%")

# ── Pairwise distances in PCA space ──────────────────────────────────────
print("Computing pairwise distances...")
behav_dist = squareform(pdist(behav_pca))
clip8_dist = squareform(pdist(clip8_pca))
alexnet_dist = squareform(pdist(alexnet_pca))
vit_dist = squareform(pdist(vit_pca))

# Only consider assigned concepts
assigned_mask = labels >= 0
assigned_idx = np.where(assigned_mask)[0]
n_assigned = len(assigned_idx)
print(f"Assigned concepts: {n_assigned}")

# Get upper-triangle pairwise distances for assigned concepts only
pairs_i, pairs_j = np.triu_indices(n_assigned, k=1)
pairs_global_i = assigned_idx[pairs_i]
pairs_global_j = assigned_idx[pairs_j]

behav_pair_dists = behav_dist[pairs_global_i, pairs_global_j]
clip8_pair_dists = clip8_dist[pairs_global_i, pairs_global_j]
alexnet_pair_dists = alexnet_dist[pairs_global_i, pairs_global_j]
vit_pair_dists = vit_dist[pairs_global_i, pairs_global_j]

# ── Distance statistics ──────────────────────────────────────────────────
print("\n" + "="*70)
print("DISTANCE STATISTICS (assigned concepts, PCA 2D space)")
print("="*70)
for name, dists in [("Behavioral", behav_pair_dists), ("CLIP-8", clip8_pair_dists),
                     ("AlexNet", alexnet_pair_dists), ("ViT", vit_pair_dists)]:
    pcts = np.percentile(dists, [5, 10, 25, 35, 50, 75, 90, 95])
    print(f"\n{name}:")
    print(f"  Mean={dists.mean():.4f}, Median={np.median(dists):.4f}")
    print(f"  5th={pcts[0]:.4f}, 10th={pcts[1]:.4f}, 25th={pcts[2]:.4f}, "
          f"35th={pcts[3]:.4f}, 50th={pcts[4]:.4f}, 75th={pcts[5]:.4f}, "
          f"90th={pcts[6]:.4f}, 95th={pcts[7]:.4f}")

# ── Filter pairs ─────────────────────────────────────────────────────────
# Different super-categories
diff_cat = labels[pairs_global_i] != labels[pairs_global_j]

# Distance thresholds
behav_p75 = np.percentile(behav_pair_dists, 75)
clip8_p50 = np.percentile(clip8_pair_dists, 50)
alexnet_p10 = np.percentile(alexnet_pair_dists, 10)
alexnet_p35 = np.percentile(alexnet_pair_dists, 35)
vit_p10 = np.percentile(vit_pair_dists, 10)
vit_p35 = np.percentile(vit_pair_dists, 35)

print(f"\nThresholds:")
print(f"  Behavioral > {behav_p75:.4f} (75th pct)")
print(f"  CLIP-8 > {clip8_p50:.4f} (50th pct)")
print(f"  AlexNet in [{alexnet_p10:.4f}, {alexnet_p35:.4f}] (10th-35th pct)")
print(f"  ViT in [{vit_p10:.4f}, {vit_p35:.4f}] (10th-35th pct)")

# Pre-filter: pairs that meet AlexNet AND ViT distance criteria + different categories
pair_mask = (
    diff_cat &
    (alexnet_pair_dists >= alexnet_p10) & (alexnet_pair_dists <= alexnet_p35) &
    (vit_pair_dists >= vit_p10) & (vit_pair_dists <= vit_p35)
)

filtered_pair_indices = np.where(pair_mask)[0]
print(f"\nFiltered pairs (AlexNet 10-35th & ViT 10-35th & diff cat): {len(filtered_pair_indices)}")

# Build adjacency for filtered pairs
from collections import defaultdict
adj = defaultdict(set)
pair_data = {}  # (gi, gj) -> pair index in original arrays

for pidx in filtered_pair_indices:
    gi, gj = pairs_global_i[pidx], pairs_global_j[pidx]
    adj[gi].add(gj)
    adj[gj].add(gi)
    pair_data[(gi, gj)] = pidx
    pair_data[(gj, gi)] = pidx

# ── Find triplets ────────────────────────────────────────────────────────
print("\nSearching for triplets...")

# For each concept in adjacency, find triplets among its neighbors
candidates = []
nodes = sorted(adj.keys())

for a in nodes:
    neighbors_a = sorted(adj[a])
    for bi in range(len(neighbors_a)):
        b = neighbors_a[bi]
        if b <= a:
            continue
        for ci in range(bi + 1, len(neighbors_a)):
            c = neighbors_a[ci]
            if c <= b:
                continue
            # Check b-c pair exists in filtered set
            if c not in adj[b]:
                continue

            # All 3 must be different super-categories
            la, lb, lc = labels[a], labels[b], labels[c]
            if la == lb or la == lc or lb == lc:
                continue

            # Compute mean distances
            trip = sorted([a, b, c])
            pairs_abc = [(trip[0], trip[1]), (trip[0], trip[2]), (trip[1], trip[2])]

            b_dists = [behav_dist[i, j] for i, j in pairs_abc]
            c8_dists = [clip8_dist[i, j] for i, j in pairs_abc]
            a_dists = [alexnet_dist[i, j] for i, j in pairs_abc]
            v_dists = [vit_dist[i, j] for i, j in pairs_abc]

            mean_b = np.mean(b_dists)
            mean_c8 = np.mean(c8_dists)
            mean_a = np.mean(a_dists)
            mean_v = np.mean(v_dists)

            # Check behavioral and clip8 thresholds on the mean
            if mean_b < behav_p75 or mean_c8 < clip8_p50:
                continue

            score = (mean_b * mean_c8) / (mean_a + mean_v + 0.01)

            candidates.append({
                "concepts": (trip[0], trip[1], trip[2]),
                "names": (concept_names[trip[0]], concept_names[trip[1]], concept_names[trip[2]]),
                "cats": (SUPER_ORDER[labels[trip[0]]], SUPER_ORDER[labels[trip[1]]], SUPER_ORDER[labels[trip[2]]]),
                "behav_mean": mean_b,
                "clip8_mean": mean_c8,
                "alexnet_mean": mean_a,
                "vit_mean": mean_v,
                "behav_dists": b_dists,
                "clip8_dists": c8_dists,
                "alexnet_dists": a_dists,
                "vit_dists": v_dists,
                "score": score,
            })

print(f"Found {len(candidates)} triplet candidates")

# Sort by score descending
candidates.sort(key=lambda x: x["score"], reverse=True)

# ── Print top 30 ─────────────────────────────────────────────────────────
print("\n" + "="*70)
print("TOP 30 TRIPLET CANDIDATES")
print("="*70)

for rank, cand in enumerate(candidates[:30], 1):
    i, j, k = cand["concepts"]
    print(f"\n{'─'*70}")
    print(f"Rank {rank} | Score: {cand['score']:.4f}")
    print(f"  Concepts: {cand['names'][0]} ({cand['cats'][0]}), "
          f"{cand['names'][1]} ({cand['cats'][1]}), "
          f"{cand['names'][2]} ({cand['cats'][2]})")
    print(f"  Category colors: {SUPER_COLORS[cand['cats'][0]]}, "
          f"{SUPER_COLORS[cand['cats'][1]]}, {SUPER_COLORS[cand['cats'][2]]}")
    print(f"  Mean distances:")
    print(f"    Behavioral: {cand['behav_mean']:.4f}  (individual: {cand['behav_dists'][0]:.4f}, {cand['behav_dists'][1]:.4f}, {cand['behav_dists'][2]:.4f})")
    print(f"    CLIP-8:     {cand['clip8_mean']:.4f}  (individual: {cand['clip8_dists'][0]:.4f}, {cand['clip8_dists'][1]:.4f}, {cand['clip8_dists'][2]:.4f})")
    print(f"    AlexNet:    {cand['alexnet_mean']:.4f}  (individual: {cand['alexnet_dists'][0]:.4f}, {cand['alexnet_dists'][1]:.4f}, {cand['alexnet_dists'][2]:.4f})")
    print(f"    ViT:        {cand['vit_mean']:.4f}  (individual: {cand['vit_dists'][0]:.4f}, {cand['vit_dists'][1]:.4f}, {cand['vit_dists'][2]:.4f})")
    print(f"  PCA coordinates:")
    for idx_c, name_c in zip(cand["concepts"], cand["names"]):
        print(f"    {name_c}:")
        print(f"      Behav: ({behav_pca[idx_c, 0]:.3f}, {behav_pca[idx_c, 1]:.3f})")
        print(f"      CLIP8: ({clip8_pca[idx_c, 0]:.3f}, {clip8_pca[idx_c, 1]:.3f})")
        print(f"      AlexNet: ({alexnet_pca[idx_c, 0]:.3f}, {alexnet_pca[idx_c, 1]:.3f})")
        print(f"      ViT: ({vit_pca[idx_c, 0]:.3f}, {vit_pca[idx_c, 1]:.3f})")

# ── Check image paths ────────────────────────────────────────────────────
print("\n" + "="*70)
print("IMAGE PATH CHECK")
print("="*70)

rep_image_paths = behav_data["rep_image_paths"]
# Check first few from top candidates
check_count = 0
missing_count = 0
for cand in candidates[:5]:
    for idx_c in cand["concepts"]:
        path = str(rep_image_paths[idx_c])
        exists = os.path.exists(path)
        if not exists:
            missing_count += 1
            # Try alternative path
            alt_path = path.replace("/home/ymehta3/.cache/bonner-datasets/", "/data/shared/datasets/")
            alt_exists = os.path.exists(alt_path)
            print(f"  MISSING: {path}")
            print(f"    Alt path ({alt_path}): {'EXISTS' if alt_exists else 'MISSING'}")
        check_count += 1

if missing_count == 0:
    print(f"All {check_count} checked image paths exist!")
else:
    print(f"{missing_count}/{check_count} images missing from original paths")

# Also show the base image directory
print(f"\nImage base: /home/ymehta3/.cache/bonner-datasets/hebart2019.things/images/object_images/")
print(f"Alt base:   /data/shared/datasets/hebart2019.things/images/object_images/")

# Quick summary of top 5
print("\n" + "="*70)
print("QUICK SUMMARY - TOP 5")
print("="*70)
print(f"{'Rank':<5} {'Score':<8} {'Behav':<8} {'CLIP8':<8} {'AlexNet':<8} {'ViT':<8} {'Concepts'}")
for rank, cand in enumerate(candidates[:5], 1):
    names_str = ", ".join(f"{n} ({c})" for n, c in zip(cand["names"], cand["cats"]))
    print(f"{rank:<5} {cand['score']:<8.3f} {cand['behav_mean']:<8.4f} {cand['clip8_mean']:<8.4f} "
          f"{cand['alexnet_mean']:<8.4f} {cand['vit_mean']:<8.4f} {names_str}")
