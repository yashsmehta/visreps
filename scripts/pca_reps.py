import os
import numpy as np
import pandas as pd
import xarray as xr
import argparse
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def get_feature_layer_name(dataset):
    """Return the appropriate feature layer name for each dataset."""
    if dataset == "cifar10":
        return "avgpool"
    else:  # tiny-imagenet or imagenet
        return "classifier.4"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', type=str, required=True,
        choices=['cifar10', 'tiny-imagenet', 'imagenet'],
        help='Dataset to process features from'
    )
    parser.add_argument(
        '--n_bits', type=int, default=2,
        help='Number of bits for classification (will create 2^n_bits classes)'
    )
    args = parser.parse_args()

    # Load features from netCDF file
    features_path = f"datasets/obj_cls/{args.dataset}/classification_features.nc"
    base_dir = os.path.dirname(features_path)
    labels_dir = os.path.join(base_dir, "pca_labels")
    os.makedirs(labels_dir, exist_ok=True)

    ds = xr.open_dataset(features_path)
    print("Dataset structure:")
    print(ds)

    # Extract image names and features
    image_names = ds.image.values
    feature_layer = get_feature_layer_name(args.dataset)
    feature_array = ds[feature_layer].values
    n_samples = feature_array.shape[0]
    features_2d = feature_array.reshape(n_samples, -1)

    print(f"\nFeature array shape: {feature_array.shape}")
    print(f"Reshaped features shape: {features_2d.shape}")

    # Standardize the features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_2d)

    # Perform PCA on the standardized features
    pca = PCA()
    pc_scores = pca.fit_transform(features_scaled)

    # Create binary classifications for the first n_bits principal components
    binary_labels = []
    for i in range(args.n_bits):
        pc_i_scores = pc_scores[:, i]
        median_pc = np.median(pc_i_scores)
        binary_labels.append((pc_i_scores > median_pc).astype(int))
    binary_labels = np.array(binary_labels).T  # shape: (n_samples, n_bits)

    # Convert binary labels (bits) to a single decimal class label (0 to 2^n_bits - 1)
    class_labels = np.zeros(n_samples, dtype=int)
    for i in range(args.n_bits):
        class_labels += binary_labels[:, i] * (2 ** (args.n_bits - 1 - i))

    print(f"\nUsing first {args.n_bits} PCs to create {2**args.n_bits} classes")
    for i in range(args.n_bits):
        explained_variance = pca.explained_variance_ratio_[i] * 100
        print(f"PC{i+1} explains {explained_variance:.2f}% of variance")

    print("\nClass distribution:")
    for i in range(2**args.n_bits):
        count = np.sum(class_labels == i)
        print(f"Class {i}: {count} samples")

    # For CIFAR-10, extract original class labels from image names
    if args.dataset == "cifar10":
        original_labels = np.array([int(name.split('_')[0]) for name in image_names])
        df = pd.DataFrame({
            'image': image_names,
            'original_label': original_labels,
            'pca_label': class_labels
        })
    else:
        df = pd.DataFrame({
            'image': image_names,
            'pca_label': class_labels
        })

    output_csv_file = os.path.join(labels_dir, f"n_classes_{2**args.n_bits}.csv")
    df.to_csv(output_csv_file, index=False)
    print(f"\nSaved PCA labels to {output_csv_file}")

    # Clean up: close the dataset
    ds.close()


if __name__ == '__main__':
    main()