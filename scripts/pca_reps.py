import os
import argparse
import numpy as np
import pandas as pd
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

    # Load features
    features_path = f"datasets/obj_cls/{args.dataset}/features.npz"
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found at {features_path}")
    
    labels_dir = os.path.join(os.path.dirname(features_path), "pca_labels")
    os.makedirs(labels_dir, exist_ok=True)

    print(f"Loading features from {features_path}")
    data_dict = np.load(features_path, allow_pickle=True)
    
    # Handle potential byte strings in image names
    image_names = data_dict['image_names']
    if isinstance(image_names[0], (bytes, np.bytes_)):
        image_names = [name.decode('utf-8') for name in image_names]
    
    # Load and verify fc2 features
    if 'fc2' not in data_dict:
        raise ValueError("fc2 features not found in the features file")
    feature_array = data_dict['fc2']
    
    # Ensure features are 2D
    n_samples = feature_array.shape[0]
    if feature_array.ndim != 2:
        features_2d = feature_array.reshape(n_samples, -1)
        print(f"Reshaped features from {feature_array.shape} to {features_2d.shape}")
    else:
        features_2d = feature_array
        print(f"Feature shape: {features_2d.shape}")

    # Standardize features and perform PCA
    print("Standardizing features...")
    features_scaled = StandardScaler().fit_transform(features_2d)
    
    print("Fitting PCA...")
    pca = PCA().fit(features_scaled)
    pc_scores = pca.transform(features_scaled)
    print(f"PCA scores shape: {pc_scores.shape}, dtype: {pc_scores.dtype}")
    print(f"Using first {n_bits} components out of {pc_scores.shape[1]} total components")

    # Create binary labels from the first n_bits principal components
    print(f"\nCreating {args.n_classes} classes using first {n_bits} PCs")
    binary_labels = np.array([
        (pc_scores[:, i] > np.median(pc_scores[:, i])).astype(int)
        for i in range(n_bits)
    ]).T
    class_labels = sum(binary_labels[:, i] * (2 ** (n_bits - 1 - i)) for i in range(n_bits))
    class_labels %= args.n_classes

    # Print statistics
    for i in range(n_bits):
        print(f"PC{i+1} explains {pca.explained_variance_ratio_[i]*100:.2f}% of variance")

    print("\nClass distribution:")
    unique_labels, counts = np.unique(class_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Class {label}: {count} samples ({count/n_samples*100:.1f}%)")

    # Save PCA labels to CSV
    df = pd.DataFrame({'image': image_names, 'pca_label': class_labels})
    output_csv = os.path.join(labels_dir, f"n_classes_{args.n_classes}.csv")
    df.to_csv(output_csv, index=False)
    print(f"\nSaved PCA labels to {output_csv}")

if __name__ == '__main__':
    main()