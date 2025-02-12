import os
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from math import log2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['tiny-imagenet', 'imagenet'],
                        help='Dataset to process features from')
    parser.add_argument('--n_classes', type=int, required=True,
                        help='Number of classes to create using PCA (must be a power of 2)')
    args = parser.parse_args()

    # Validate that n_classes is a positive power of 2.
    if args.n_classes <= 0 or (args.n_classes & (args.n_classes - 1)) != 0:
        raise ValueError(f"n_classes must be a power of 2 (2^k), got {args.n_classes}")
    n_bits = int(log2(args.n_classes))

    # Load features from the NetCDF file.
    features_path = f"datasets/obj_cls/{args.dataset}/classification_features.nc"
    labels_dir = os.path.join(os.path.dirname(features_path), "pca_labels")
    os.makedirs(labels_dir, exist_ok=True)

    ds = xr.open_dataset(features_path)
    print("Dataset structure:")
    print(ds)

    image_names = ds.image.values
    feature_array = ds["classifier.5"].values
    n_samples = feature_array.shape[0]
    features_2d = feature_array.reshape(n_samples, -1)
    print(f"\nFeatures shape: {feature_array.shape} -> reshaped to {features_2d.shape}")

    # Standardize features and perform PCA.
    features_scaled = StandardScaler().fit_transform(features_2d)
    print("Fitting PCA...")
    pca = PCA().fit(features_scaled)
    pc_scores = pca.transform(features_scaled)

    # Create binary labels from the first n_bits principal components.
    binary_labels = np.array([
        (pc_scores[:, i] > np.median(pc_scores[:, i])).astype(int)
        for i in range(n_bits)
    ]).T
    class_labels = sum(binary_labels[:, i] * (2 ** (n_bits - 1 - i)) for i in range(n_bits))
    class_labels %= args.n_classes

    print(f"\nUsing first {n_bits} PCs to create {args.n_classes} classes")
    for i in range(n_bits):
        print(f"PC{i+1} explains {pca.explained_variance_ratio_[i]*100:.2f}% of variance")

    print("\nClass distribution:")
    for i in range(args.n_classes):
        print(f"Class {i}: {(class_labels == i).sum()} samples")

    # Save PCA labels to CSV.
    df = pd.DataFrame({'image': image_names, 'pca_label': class_labels})
    output_csv = os.path.join(labels_dir, f"n_classes_{args.n_classes}.csv")
    df.to_csv(output_csv, index=False)
    print(f"\nSaved PCA labels to {output_csv}")
    
    ds.close()

if __name__ == '__main__':
    main()