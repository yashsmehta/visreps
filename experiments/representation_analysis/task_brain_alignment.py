"""
Task-Brain Alignment Analysis

Investigates whether dimensions that are important for task classification
(discriminating training classes) are the same dimensions that predict neural responses.

Key Question: At the "sweet spot" granularity (e.g., 32 classes), does what the model
learns to discriminate align with what the brain represents?

Analysis:
1. Task-Discriminative Dimensions: Which representation dimensions separate training classes?
   - Compute Fisher's Linear Discriminant ratio per dimension
   - Or: Use class centroids and measure between-class vs within-class variance

2. Brain-Predictive Dimensions: Which dimensions predict neural responses?
   - Fit ridge regression encoding model
   - Look at regression weight magnitudes

3. Alignment: Cosine similarity between task-importance and brain-importance vectors
"""

import os
import sys

# Setup path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, '.env'))

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import pandas as pd

from torchvision import transforms
import torchvision.models as models

from himalaya.backend import set_backend
from himalaya.ridge import RidgeCV

from visreps.models.utils import FeatureExtractor
from visreps.dataloaders.obj_cls import get_obj_cls_loader
from visreps.dataloaders.neural import load_nsd_data
from visreps.analysis.encoding_score import _pearson_r


# =============================================================================
# Configuration
# =============================================================================
# Model checkpoint for 32-class trained model (local path)
CHECKPOINT_32WAY = os.path.join(PROJECT_ROOT, "model_checkpoints/global_pca_classes/cfg32a/checkpoint_epoch_20.pth")

# PCA labels for the 32-class model (must match training folder)
PCA_LABELS_PATH = os.path.join(PROJECT_ROOT, "pca_labels/pca_labels_alexnet_global/n_classes_32.csv")

# Neural data config
NEURAL_CFG = {
    "region": "EVC",  # Early Visual Cortex - where you see the effect
    "subject_idx": 0,
}

# Analysis settings
LAYERS = ["conv2", "conv3", "conv4", "conv5", "fc1", "fc2"]  # Layers to analyze
SEED = 42
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "experiments/representation_analysis")


# =============================================================================
# Model Loading
# =============================================================================
def load_pretrained_alexnet(device):
    """Load pretrained ImageNet AlexNet."""
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    return model.to(device).eval()


def load_checkpoint_model(checkpoint_path, device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    return checkpoint['model'].to(device).eval()


# =============================================================================
# Feature Extraction
# =============================================================================
def extract_features(model, images, device, layer):
    """Extract features from a specific layer for a batch of images."""
    extractor = FeatureExtractor(model, return_nodes={layer: layer})
    extractor.to(device).eval()

    with torch.no_grad():
        feats = extractor(images.to(device))

    # Flatten spatial dimensions
    feat = feats[layer]
    if feat.dim() > 2:
        feat = feat.flatten(start_dim=1)

    return feat.cpu().numpy()


def extract_features_from_loader(model, loader, device, layer):
    """Extract features for all images in a dataloader."""
    extractor = FeatureExtractor(model, return_nodes={layer: layer})
    extractor.to(device).eval()

    features = []
    with torch.no_grad():
        for images, _ in tqdm(loader, desc=f"Extracting {layer}", leave=False):
            feats = extractor(images.to(device))
            feat = feats[layer]
            if feat.dim() > 2:
                feat = feat.flatten(start_dim=1)
            features.append(feat.cpu().numpy())

    return np.vstack(features)


# =============================================================================
# Task-Discriminative Dimensions (Fisher's Linear Discriminant)
# =============================================================================
def compute_fisher_discriminant_per_dim(features, labels):
    """
    Compute Fisher's Linear Discriminant ratio for each feature dimension.

    FLD ratio = between-class variance / within-class variance

    Higher ratio = dimension is more discriminative for the task.

    Args:
        features: (n_samples, n_features) array
        labels: (n_samples,) array of class labels

    Returns:
        fld_scores: (n_features,) array of FLD ratios per dimension
    """
    n_samples, n_features = features.shape
    classes = np.unique(labels)
    n_classes = len(classes)

    # Global mean per dimension
    global_mean = features.mean(axis=0)

    # Compute between-class and within-class variance per dimension
    between_var = np.zeros(n_features)
    within_var = np.zeros(n_features)

    for c in classes:
        class_mask = labels == c
        class_features = features[class_mask]
        n_c = class_features.shape[0]

        class_mean = class_features.mean(axis=0)

        # Between-class: weighted squared distance from class mean to global mean
        between_var += n_c * (class_mean - global_mean) ** 2

        # Within-class: variance within this class
        within_var += ((class_features - class_mean) ** 2).sum(axis=0)

    # Normalize
    between_var /= n_samples
    within_var /= n_samples

    # FLD ratio (add small epsilon for numerical stability)
    fld_scores = between_var / (within_var + 1e-10)

    return fld_scores


def compute_class_centroid_importance(features, labels):
    """
    Alternative: Compute importance based on class centroid separation.

    For each dimension, compute the variance of class centroids (how spread out
    the classes are along this dimension).

    Args:
        features: (n_samples, n_features) array
        labels: (n_samples,) array of class labels

    Returns:
        importance: (n_features,) array of centroid variance per dimension
    """
    classes = np.unique(labels)
    centroids = np.array([features[labels == c].mean(axis=0) for c in classes])

    # Variance of centroids along each dimension
    centroid_variance = centroids.var(axis=0)

    return centroid_variance


# =============================================================================
# Brain-Predictive Dimensions (using himalaya - consistent with main pipeline)
# =============================================================================
def compute_brain_predictive_weights(features, neural_responses, seed=SEED):
    """
    Fit ridge regression using himalaya (matches visreps/analysis/encoding_score.py).
    Returns per-feature importance weights based on regression coefficients.

    Args:
        features: (n_samples, n_features) numpy array
        neural_responses: (n_samples, n_voxels) numpy array
        seed: random seed for train/test split

    Returns:
        weights: (n_features,) array - mean |weight| per feature across voxels
        mean_r: float - mean Pearson r across voxels (encoding score)
        alpha_median: float - median selected alpha across voxels
    """
    backend = set_backend("torch_cuda", on_error="warn")

    alphas = np.logspace(-10, 10, 20)
    n_folds = 5
    train_fraction = 0.8

    # Train/test split
    n = len(features)
    idx = np.random.default_rng(seed).permutation(n)
    split = int(train_fraction * n)
    tr_idx, te_idx = idx[:split], idx[split:]

    # Convert to backend tensors
    X = backend.asarray(torch.from_numpy(features).float())
    Y = backend.asarray(torch.from_numpy(neural_responses).float())

    X_tr, X_te = X[tr_idx], X[te_idx]
    Y_tr, Y_te = Y[tr_idx], Y[te_idx]

    # Z-normalize using training statistics only (prevents data leakage)
    train_mean = X_tr.mean(dim=0)
    train_std = X_tr.std(dim=0) + 1e-8
    X_tr = (X_tr - train_mean) / train_std
    X_te = (X_te - train_mean) / train_std

    # Fit ridge with per-voxel alpha selection
    model = RidgeCV(alphas=alphas, cv=n_folds, fit_intercept=True)
    model.fit(X_tr, Y_tr)

    # Predictions and score
    pred = backend.asarray(model.predict(X_te))
    mean_r = float(_pearson_r(pred, Y_te).mean().cpu())

    # Extract weights: himalaya coef_ shape is (n_voxels, n_features)
    coef = backend.to_numpy(model.coef_)
    weights = np.abs(coef).mean(axis=0)

    # Alpha info
    alpha_median = float(np.median(backend.to_numpy(model.best_alphas_)))

    return weights, mean_r, alpha_median


# =============================================================================
# NSD Feature Extraction and Encoding
# =============================================================================
def extract_nsd_and_fit_encoding(model, layer, nsd_stimuli, nsd_neural_data, device):
    """
    Extract features for NSD stimuli and fit encoding model.

    Returns:
        nsd_features: (n_samples, n_features) array
        brain_weights: (n_features,) array of brain importance weights
        mean_r: Mean voxel correlation (encoding score)
        alpha_median: Median regularization strength across voxels
    """
    from PIL import Image

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    extractor = FeatureExtractor(model, return_nodes={layer: layer})
    extractor.to(device).eval()

    nsd_features = []
    nsd_neural = []
    nsd_ids = sorted(nsd_stimuli.keys())

    with torch.no_grad():
        for stim_id in tqdm(nsd_ids, desc=f"      NSD", leave=False):
            img_data = nsd_stimuli[stim_id]
            if isinstance(img_data, np.ndarray):
                img = Image.fromarray(img_data.astype('uint8'), 'RGB')
            else:
                img = img_data
            img_tensor = transform(img).unsqueeze(0).to(device)

            feats = extractor(img_tensor)
            feat = feats[layer]
            if feat.dim() > 2:
                feat = feat.flatten(start_dim=1)
            nsd_features.append(feat.cpu().numpy().squeeze())
            nsd_neural.append(nsd_neural_data[stim_id])

    nsd_features = np.array(nsd_features)
    nsd_neural = np.array(nsd_neural)

    # Fit encoding model (himalaya with per-voxel alpha)
    brain_weights, mean_r, alpha_median = compute_brain_predictive_weights(nsd_features, nsd_neural)

    return nsd_features, brain_weights, mean_r, alpha_median


# =============================================================================
# Alignment Computation
# =============================================================================
def compute_alignment(task_weights, brain_weights):
    """
    Compute alignment between task-discriminative and brain-predictive dimensions.

    Returns multiple alignment metrics:
    - Cosine similarity
    - Spearman correlation (rank-based)
    - Overlap of top-K dimensions
    """
    # Normalize for cosine similarity
    task_norm = task_weights / (np.linalg.norm(task_weights) + 1e-10)
    brain_norm = brain_weights / (np.linalg.norm(brain_weights) + 1e-10)

    cosine_sim = np.dot(task_norm, brain_norm)

    # Spearman correlation (rank-based)
    spearman_r, spearman_p = spearmanr(task_weights, brain_weights)

    # Pearson correlation
    pearson_r, pearson_p = pearsonr(task_weights, brain_weights)

    # Top-K overlap
    overlaps = {}
    for k in [100, 500, 1000]:
        if k > len(task_weights):
            k = len(task_weights) // 2
        top_task = set(np.argsort(task_weights)[-k:])
        top_brain = set(np.argsort(brain_weights)[-k:])
        overlap = len(top_task & top_brain) / k
        overlaps[f'top_{k}_overlap'] = overlap

    return {
        'cosine_similarity': cosine_sim,
        'spearman_r': spearman_r,
        'spearman_p': spearman_p,
        'pearson_r': pearson_r,
        'pearson_p': pearson_p,
        **overlaps
    }


# =============================================================================
# Visualization
# =============================================================================
def plot_task_brain_alignment(task_weights, brain_weights, alignment_metrics,
                               model_name, layer, output_path):
    """
    Create visualization of task-brain alignment.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Scatter plot of task vs brain weights
    ax1 = axes[0, 0]
    ax1.scatter(task_weights, brain_weights, alpha=0.3, s=5)
    ax1.set_xlabel('Task-Discriminative Weight (FLD)', fontsize=11)
    ax1.set_ylabel('Brain-Predictive Weight', fontsize=11)
    ax1.set_title(f'Task vs Brain Dimension Importance\n'
                  f'Spearman r = {alignment_metrics["spearman_r"]:.3f} '
                  f'(p = {alignment_metrics["spearman_p"]:.2e})',
                  fontsize=12, fontweight='bold')
    ax1.set_facecolor('#FAFAFA')

    # Add trend line
    z = np.polyfit(task_weights, brain_weights, 1)
    p = np.poly1d(z)
    x_line = np.linspace(task_weights.min(), task_weights.max(), 100)
    ax1.plot(x_line, p(x_line), 'r--', linewidth=2, label='Linear fit')
    ax1.legend()

    # Plot 2: Ranked comparison
    ax2 = axes[0, 1]
    task_ranks = np.argsort(np.argsort(task_weights))
    brain_ranks = np.argsort(np.argsort(brain_weights))
    ax2.scatter(task_ranks, brain_ranks, alpha=0.2, s=3)
    ax2.plot([0, len(task_ranks)], [0, len(task_ranks)], 'r--', linewidth=2)
    ax2.set_xlabel('Task Importance Rank', fontsize=11)
    ax2.set_ylabel('Brain Importance Rank', fontsize=11)
    ax2.set_title('Rank Comparison\n(Perfect alignment = diagonal)',
                  fontsize=12, fontweight='bold')
    ax2.set_facecolor('#FAFAFA')

    # Plot 3: Distribution of weights
    ax3 = axes[1, 0]
    ax3.hist(task_weights, bins=50, alpha=0.6, label='Task-Discriminative',
             color='#1f77b4', density=True)
    ax3.hist(brain_weights, bins=50, alpha=0.6, label='Brain-Predictive',
             color='#ff7f0e', density=True)
    ax3.set_xlabel('Weight Magnitude', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.set_title('Distribution of Dimension Importance', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.set_facecolor('#FAFAFA')

    # Plot 4: Top-K overlap bar chart
    ax4 = axes[1, 1]
    overlap_keys = [k for k in alignment_metrics if 'overlap' in k]
    overlap_values = [alignment_metrics[k] for k in overlap_keys]
    overlap_labels = [k.replace('top_', 'Top ').replace('_overlap', '') for k in overlap_keys]

    bars = ax4.bar(overlap_labels, overlap_values, color='#2ca02c', alpha=0.8)
    ax4.axhline(y=0.5, color='gray', linestyle='--', label='Random chance (if K = 50% of dims)')
    ax4.set_xlabel('Number of Top Dimensions', fontsize=11)
    ax4.set_ylabel('Overlap Fraction', fontsize=11)
    ax4.set_title('Overlap of Top-K Important Dimensions', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, 1)
    ax4.set_facecolor('#FAFAFA')

    # Add values on bars
    for bar, val in zip(bars, overlap_values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle(f'Task-Brain Alignment Analysis\n{model_name} - {layer}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


def plot_weight_comparison_heatmap(task_weights, brain_weights, output_path, n_dims=100):
    """
    Create a heatmap showing top dimensions by task vs brain importance.
    """
    # Get top dimensions by each criterion
    top_task_idx = np.argsort(task_weights)[-n_dims:][::-1]
    top_brain_idx = np.argsort(brain_weights)[-n_dims:][::-1]

    # Create a combined set of interesting dimensions
    interesting_idx = np.unique(np.concatenate([top_task_idx[:50], top_brain_idx[:50]]))

    fig, ax = plt.subplots(figsize=(12, 8))

    # Create data for heatmap
    data = np.column_stack([
        task_weights[interesting_idx],
        brain_weights[interesting_idx]
    ])

    # Normalize each column for visualization
    data_norm = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-10)

    # Sort by task weight
    sort_idx = np.argsort(data_norm[:, 0])[::-1]
    data_norm = data_norm[sort_idx]

    im = ax.imshow(data_norm.T, aspect='auto', cmap='viridis')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Task-Discriminative', 'Brain-Predictive'])
    ax.set_xlabel('Feature Dimensions (sorted by task importance)', fontsize=11)
    ax.set_title('Comparison of Top Dimensions\n(Normalized importance)',
                 fontsize=13, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Normalized Importance')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


# =============================================================================
# Multi-Layer Visualization
# =============================================================================
def plot_alignment_across_layers(results_df, output_path):
    """
    Plot task-brain alignment across layers comparing models.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    layers = results_df['layer'].unique()
    x = np.arange(len(layers))

    for model_name in results_df['model'].unique():
        model_data = results_df[results_df['model'] == model_name]

        # Ensure correct order
        spearman_r = [model_data[model_data['layer'] == l]['spearman_r'].values[0] for l in layers]
        cosine_sim = [model_data[model_data['layer'] == l]['cosine_sim'].values[0] for l in layers]
        encoding_r = [model_data[model_data['layer'] == l]['encoding_r'].values[0] for l in layers]

        marker = 'o-' if '32' in model_name else 's--'

        axes[0].plot(x, spearman_r, marker, linewidth=2, markersize=8, label=model_name)
        axes[1].plot(x, cosine_sim, marker, linewidth=2, markersize=8, label=model_name)
        axes[2].plot(x, encoding_r, marker, linewidth=2, markersize=8, label=model_name)

    axes[0].set_ylabel('Task-Brain Alignment (Spearman r)', fontsize=11)
    axes[0].set_title('Task-Brain Dimension Alignment', fontsize=12, fontweight='bold')

    axes[1].set_ylabel('Task-Brain Alignment (Cosine)', fontsize=11)
    axes[1].set_title('Task-Brain Cosine Similarity', fontsize=12, fontweight='bold')

    axes[2].set_ylabel('Encoding Score (Mean r)', fontsize=11)
    axes[2].set_title('Neural Encoding Performance', fontsize=12, fontweight='bold')

    for ax in axes:
        ax.set_xlabel('Layer', fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(layers)
        ax.legend(fontsize=9)
        ax.set_facecolor('#FAFAFA')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Task-Brain Alignment Across Layers: 32-class vs Pretrained',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {output_path}")
    plt.close()


# =============================================================================
# Main Analysis
# =============================================================================
def main():
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # =========================================================================
    # Step 1: Load models
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 1: Loading models")
    print("=" * 60)

    model_32way = load_checkpoint_model(CHECKPOINT_32WAY, device)
    print(f"Loaded 32-class model from: {CHECKPOINT_32WAY}")

    model_pretrained = load_pretrained_alexnet(device)
    print("Loaded pretrained AlexNet (1000-class) for comparison")

    # =========================================================================
    # Step 2: Load ImageNet data with 32-class PCA labels
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 2: Loading ImageNet data with 32-class labels")
    print("=" * 60)

    cfg_imagenet = {"dataset": "imagenet-mini-50", "batchsize": 256, "num_workers": 8}
    _, loaders = get_obj_cls_loader(cfg_imagenet, shuffle=False, train_test_split=False)
    imagenet_loader = loaders['all']
    print(f"ImageNet images: {len(imagenet_loader.dataset)}")

    # Load 32-class labels
    df_labels = pd.read_csv(PCA_LABELS_PATH)
    label_map = dict(zip(df_labels['image'], df_labels['pca_label']))

    # Get labels for images in loader and create filtered version
    imagenet_labels_all = []
    valid_indices = []
    for i, sample in enumerate(imagenet_loader.dataset.samples):
        img_name = sample[2]  # Filename
        label = label_map.get(img_name, -1)
        imagenet_labels_all.append(label)
        if label >= 0:
            valid_indices.append(i)

    imagenet_labels_all = np.array(imagenet_labels_all)
    valid_mask = imagenet_labels_all >= 0
    print(f"Valid images with labels: {valid_mask.sum()}/{len(imagenet_labels_all)}")

    # =========================================================================
    # Step 3: Load NSD neural data
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 3: Loading NSD neural data")
    print("=" * 60)

    nsd_neural_data, nsd_stimuli = load_nsd_data(NEURAL_CFG)
    print(f"NSD stimuli: {len(nsd_stimuli)}")
    print(f"Neural data shape: {next(iter(nsd_neural_data.values())).shape}")
    print(f"Brain region: {NEURAL_CFG['region']}")

    # =========================================================================
    # Step 4: Run analysis across layers
    # =========================================================================
    all_results = []
    labels_filtered = imagenet_labels_all[valid_mask]

    for layer in LAYERS:
        print(f"\n{'='*60}")
        print(f"Analyzing layer: {layer}")
        print(f"{'='*60}")

        # --- 32-class model ---
        print(f"\n  [32-class model]")

        # Extract ImageNet features
        print(f"    Extracting ImageNet features...")
        feats_32way = extract_features_from_loader(model_32way, imagenet_loader, device, layer)
        feats_32way_filtered = feats_32way[valid_mask]

        # Compute task-discriminative weights
        print(f"    Computing task weights (FLD)...")
        task_weights_32 = compute_fisher_discriminant_per_dim(feats_32way_filtered, labels_filtered)

        # Extract NSD features and compute brain weights
        print(f"    Extracting NSD features and fitting encoding model...")
        nsd_feats_32, brain_weights_32, enc_r_32, alpha_32 = extract_nsd_and_fit_encoding(
            model_32way, layer, nsd_stimuli, nsd_neural_data, device
        )

        # Compute alignment
        alignment_32 = compute_alignment(task_weights_32, brain_weights_32)
        print(f"    Task-Brain Spearman r: {alignment_32['spearman_r']:.4f}")

        all_results.append({
            'layer': layer,
            'model': '32-class',
            'task_weights': task_weights_32,
            'brain_weights': brain_weights_32,
            'alignment': alignment_32,
            'encoding_mean_r': enc_r_32,
            'alpha_median': alpha_32,
            'n_features': len(task_weights_32)
        })

        # --- Pretrained model ---
        print(f"\n  [Pretrained (1000) model]")

        # Extract ImageNet features
        print(f"    Extracting ImageNet features...")
        feats_pre = extract_features_from_loader(model_pretrained, imagenet_loader, device, layer)
        feats_pre_filtered = feats_pre[valid_mask]

        # Compute task-discriminative weights (using same 32-class labels for fair comparison)
        print(f"    Computing task weights (FLD)...")
        task_weights_pre = compute_fisher_discriminant_per_dim(feats_pre_filtered, labels_filtered)

        # Extract NSD features and compute brain weights
        print(f"    Extracting NSD features and fitting encoding model...")
        nsd_feats_pre, brain_weights_pre, enc_r_pre, alpha_pre = extract_nsd_and_fit_encoding(
            model_pretrained, layer, nsd_stimuli, nsd_neural_data, device
        )

        # Compute alignment
        alignment_pre = compute_alignment(task_weights_pre, brain_weights_pre)
        print(f"    Task-Brain Spearman r: {alignment_pre['spearman_r']:.4f}")

        all_results.append({
            'layer': layer,
            'model': 'Pretrained (1000)',
            'task_weights': task_weights_pre,
            'brain_weights': brain_weights_pre,
            'alignment': alignment_pre,
            'encoding_mean_r': enc_r_pre,
            'alpha_median': alpha_pre,
            'n_features': len(task_weights_pre)
        })

    # =========================================================================
    # Step 5: Compile results and visualize
    # =========================================================================
    print("\n" + "=" * 60)
    print("Step 5: Compiling results")
    print("=" * 60)

    results_df = pd.DataFrame([{
        'layer': r['layer'],
        'model': r['model'],
        'spearman_r': r['alignment']['spearman_r'],
        'cosine_sim': r['alignment']['cosine_similarity'],
        'pearson_r': r['alignment']['pearson_r'],
        'encoding_r': r['encoding_mean_r'],
        'alpha_median': r['alpha_median'],
        'n_features': r['n_features']
    } for r in all_results])

    print("\nResults Summary:")
    print(results_df.to_string(index=False))

    # Save results
    results_path = os.path.join(OUTPUT_DIR, "task_brain_alignment_results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"\nSaved results to: {results_path}")

    # Plot across layers
    plot_alignment_across_layers(
        results_df,
        os.path.join(OUTPUT_DIR, "task_brain_alignment_across_layers.png")
    )

    # Create detailed plot for a single layer (conv3 - typical early visual layer)
    detail_layer = "conv3"
    for r in all_results:
        if r['layer'] == detail_layer and r['model'] == "32-class":
            plot_task_brain_alignment(
                r['task_weights'], r['brain_weights'], r['alignment'],
                model_name="32-class AlexNet",
                layer=detail_layer,
                output_path=os.path.join(OUTPUT_DIR, f"task_brain_alignment_{detail_layer}_detail.png")
            )
            break

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)

    print("\nKey Findings:")
    print("-" * 40)
    for layer in LAYERS:
        r32 = results_df[(results_df['layer'] == layer) & (results_df['model'] == '32-class')]
        rpre = results_df[(results_df['layer'] == layer) & (results_df['model'] == 'Pretrained (1000)')]
        if len(r32) > 0 and len(rpre) > 0:
            diff = r32['spearman_r'].values[0] - rpre['spearman_r'].values[0]
            print(f"{layer}: 32-class={r32['spearman_r'].values[0]:.3f}, "
                  f"Pretrained={rpre['spearman_r'].values[0]:.3f}, Δ={diff:+.3f}")

    print(f"\nOutput files saved to: {OUTPUT_DIR}/")

    return results_df, all_results


if __name__ == "__main__":
    results_df, all_results = main()
