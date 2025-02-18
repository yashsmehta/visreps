import os
import argparse
import numpy as np
import pandas as pd
from math import log2
from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['tiny-imagenet', 'imagenet'],
                        help='Dataset to process features from')
    parser.add_argument('--n_classes', type=int, required=True,
                        help='Number of classes to create using PCA (must be a power of 2)')
    parser.add_argument('--batch_size', type=int, default=100000,
                        help='Batch size for incremental PCA')
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
    
    # Load and process image names
    image_names = data_dict['image_names']
    if isinstance(image_names[0], (bytes, np.bytes_)):
        image_names = [name.decode('utf-8') for name in image_names]
    
    # Use just the filenames for consistency with dataset
    image_names = [os.path.basename(name) for name in image_names]
    
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
    
    # Standardize features
    print("Standardizing features...")
    features_scaled = StandardScaler().fit_transform(features_2d)
    
    # Use IncrementalPCA with batch processing
    print(f"Fitting Incremental PCA with {n_bits} components...")
    ipca = IncrementalPCA(n_components=n_bits)
    
    # Process in batches
    batch_size = args.batch_size
    for i in range(0, n_samples, batch_size):
        batch = features_scaled[i:i + batch_size]
        ipca.partial_fit(batch)
        if (i + batch_size) % (5 * batch_size) == 0:
            print(f"Processed {i + batch_size}/{n_samples} samples")
    
    # Transform in batches
    pc_scores = np.zeros((n_samples, n_bits))
    for i in range(0, n_samples, batch_size):
        batch = features_scaled[i:i + batch_size]
        pc_scores[i:i + batch_size] = ipca.transform(batch)
    
    # Create binary labels from the principal components
    print(f"Creating {args.n_classes} classes using {n_bits} PCs")
    binary_labels = np.array([
        (pc_scores[:, i] > np.median(pc_scores[:, i])).astype(int)
        for i in range(n_bits)
    ]).T
    class_labels = sum(binary_labels[:, i] * (2 ** (n_bits - 1 - i)) for i in range(n_bits))
    class_labels %= args.n_classes

    # Print statistics
    print("\nVariance explained:")
    total_var_explained = 0
    for i in range(n_bits):
        var_explained = ipca.explained_variance_ratio_[i] * 100
        total_var_explained += var_explained
        print(f"PC{i+1}: {var_explained:.1f}%")
    print(f"Total variance explained: {total_var_explained:.1f}%")

    print("\nClass distribution:")
    unique_labels, counts = np.unique(class_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Class {label}: {count} samples ({count/n_samples*100:.1f}%)")

    # Save PCA labels to CSV
    df = pd.DataFrame({'image': image_names, 'pca_label': class_labels})
    output_csv = os.path.join(labels_dir, f"n_classes_{args.n_classes}.csv")
    df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(df)} PCA labels to {output_csv}")

if __name__ == '__main__':
    main()