import os
import numpy as np
import xarray as xr
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from visreps.dataloaders.obj_cls import get_obj_cls_loader


def get_feature_layer_name(dataset: str) -> str:
    """Return the appropriate feature layer name for the given dataset."""
    if dataset == "cifar10":
        return "avgpool"
    # Assume tiny-imagenet or imagenet
    return "classifier.4"


def compute_eigenspectrum(data_2d: np.ndarray):
    """
    Compute PCA eigenspectrum on a 2D data matrix using PyTorch SVD.
    
    Returns:
        explained_variance_ratio: numpy array of variance ratios.
        cumulative_variance_ratio: numpy array of cumulative variance ratios.
    """
    # Standardize data on CPU
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_2d)

    # Convert to PyTorch tensor (and to GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_tensor = torch.from_numpy(data_scaled).to(device)

    n_samples = data_2d.shape[0]
    try:
        if n_samples < data_2d.shape[1]:
            # Compute SVD on the covariance matrix when samples < features
            cov = torch.mm(data_tensor, data_tensor.t())
            U, S, _ = torch.linalg.svd(cov, full_matrices=False)
            S = torch.sqrt(S)  # Convert eigenvalues to singular values
            explained_variance = (S ** 2) / (n_samples - 1)
        else:
            # Regular SVD on the data tensor when samples >= features
            U, S, Vt = torch.linalg.svd(data_tensor, full_matrices=False)
            explained_variance = (S ** 2) / (n_samples - 1)
    except RuntimeError as e:
        print("SVD didn't converge, trying with double precision...")
        data_tensor = data_tensor.double()
        if n_samples < data_2d.shape[1]:
            cov = torch.mm(data_tensor, data_tensor.t())
            U, S, _ = torch.linalg.svd(cov, full_matrices=False)
            S = torch.sqrt(S)
            explained_variance = (S ** 2) / (n_samples - 1)
        else:
            U, S, Vt = torch.linalg.svd(data_tensor, full_matrices=False)
            explained_variance = (S ** 2) / (n_samples - 1)

    total_variance = explained_variance.sum()
    explained_variance_ratio = (explained_variance / total_variance).cpu().numpy()
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

    # Clean up GPU memory if needed
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return explained_variance_ratio, cumulative_variance_ratio


def plot_eigenspectra(raw_spectrum, feature_spectrum, save_path=None):
    """
    Plot the eigenspectra comparison between raw images and extracted features on a log-log scale.
    
    Note: cumulative variance ratios are not plotted in this function.
    """
    # Set style with better aesthetics
    sns.set_style("whitegrid", {"grid.linestyle": ":"})
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'legend.frameon': True,
        'legend.framealpha': 0.8,
        'legend.edgecolor': '0.8'
    })
    
    fig, ax = plt.subplots(figsize=(10, 7))

    # x-axis: component indices (starting at 1)
    x_raw = np.arange(1, len(raw_spectrum) + 1)
    x_feat = np.arange(1, len(feature_spectrum) + 1)

    # Plot with enhanced styling
    ax.loglog(x_raw, raw_spectrum, color='#1f77b4', label="Raw Images", 
              alpha=0.8, linewidth=2, zorder=2)
    ax.loglog(x_feat, feature_spectrum, color='#d62728', label="Features", 
              alpha=0.8, linewidth=2, zorder=2)

    # Set y-axis limit to 10^-7
    ax.set_ylim(bottom=1e-7)
    
    # Customize grid
    ax.grid(True, which="major", linestyle=":", color="gray", alpha=0.5, zorder=1)
    ax.grid(True, which="minor", linestyle=":", color="gray", alpha=0.2, zorder=1)

    # Labels and title with LaTeX formatting
    ax.set_xlabel("Principal Component Rank", fontsize=14)
    ax.set_ylabel("Normalized Eigenvalue (λᵢ/Σλ)", fontsize=14)
    ax.set_title("Eigenspectrum Comparison", fontsize=16, pad=15)

    # Enhance legend
    legend = ax.legend(frameon=True, fancybox=True, shadow=True, 
                      loc='upper right', bbox_to_anchor=(0.98, 0.98))
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)

    # Add light gray background to plot area
    ax.set_facecolor('#f8f9fa')
    
    # Adjust layout and save
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor='white')
        print(f"Saved plot to {save_path}")
    plt.close()


def main():
    # Configuration
    dataset = "tiny-imagenet"
    cfg = {
        "dataset": dataset,
        "dataset_path": "/data/shared/datasets/tiny-imagenet",
        "batchsize": 128,
        "num_workers": 4,
        "data_augment": False
    }

    # Create output directory
    os.makedirs("tests/outputs", exist_ok=True)

    # Load raw image data
    print("Loading raw image data...")
    datasets, _ = get_obj_cls_loader(cfg, pca_labels=False)  # Disable PCA labels
    train_dataset = datasets["train"]

    # Sample a subset of images for PCA (to manage memory)
    n_samples = 10000
    indices = torch.randperm(len(train_dataset))[:n_samples].tolist()

    # Get raw images, flatten each image to 1D vector
    raw_images = []
    for idx in indices:
        img, label = train_dataset[idx]
        raw_images.append(img.flatten().numpy())
    raw_images = np.stack(raw_images)
    print(f"Raw image data shape: {raw_images.shape}")

    # Load extracted features from NetCDF file
    print("\nLoading extracted features...")
    feature_path = f"datasets/obj_cls/{dataset}/classification_features.nc"
    ds = xr.open_dataset(feature_path)
    feature_layer = get_feature_layer_name(dataset)
    feature_array = ds[feature_layer].values

    # Index the features using the same indices and reshape to 2D
    features_2d = feature_array[indices].reshape(n_samples, -1)
    print(f"Feature data shape: {features_2d.shape}")

    # Compute eigenspectra
    print("\nComputing eigenspectra...")
    raw_spectrum, raw_cumulative = compute_eigenspectrum(raw_images)
    feature_spectrum, feature_cumulative = compute_eigenspectrum(features_2d)

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Raw Images: {len(raw_spectrum)} components")
    print(f"Features: {len(feature_spectrum)} components")

    # Plot results
    save_path = "tests/outputs/eigenspectra_comparison.png"
    plot_eigenspectra(raw_spectrum, feature_spectrum, save_path)

    # Clean up
    ds.close()


if __name__ == "__main__":
    main()