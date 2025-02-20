import os
import argparse
import numpy as np
import pandas as pd
from math import log2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='tiny-imagenet',
                        choices=['tiny-imagenet', 'imagenet'],
                        help='Dataset to process features from (default: tiny-imagenet)')
    args = parser.parse_args()

    # Use first n_bits PCA components (i.e. n_bits=6)
    n_bits = 6 

    # Load features.
    features_path = f"datasets/obj_cls/{args.dataset}/features.npz"
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found at {features_path}")
    labels_dir = os.path.join(os.path.dirname(features_path), "pca_labels")
    os.makedirs(labels_dir, exist_ok=True)
    print(f"Loading features from {features_path}")
    data_dict = np.load(features_path, allow_pickle=True)

    # Process image names.
    image_names = data_dict['image_names']
    if isinstance(image_names[0], (bytes, np.bytes_)):
        image_names = [name.decode('utf-8') for name in image_names]
    image_names = [os.path.basename(name) for name in image_names]

    # Verify and reshape features if necessary.
    if 'fc2' not in data_dict:
        raise ValueError("fc2 features not found in the features file")
    feature_array = data_dict['fc2']
    n_samples = feature_array.shape[0]
    if feature_array.ndim != 2:
        features_2d = feature_array.reshape(n_samples, -1)
        print(f"Reshaped features from {feature_array.shape} to {features_2d.shape}")
    else:
        features_2d = feature_array

    # Standardize features.
    print("Standardizing features...")
    features_scaled = StandardScaler().fit_transform(features_2d)

    # Fit PCA on 110K random samples (or all if fewer than 100K).
    n_fit = min(110000, n_samples)
    np.random.seed(42)  # For reproducibility.
    fit_indices = np.random.choice(n_samples, n_fit, replace=False)
    sample_data = features_scaled[fit_indices]
    print(f"Fitting PCA with {n_bits} components on {n_fit} random samples...")
    pca = PCA(n_components=n_bits)
    pca.fit(sample_data)

    # Print PCA statistics
    print("\nPCA Statistics:")
    print("Explained variance ratios:")
    for i, var_ratio in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i+1}: {var_ratio:.4f} ({var_ratio*100:.2f}%)")
    print(f"\nTotal variance explained: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")
    print(f"Singular values:", pca.singular_values_)

    # Transform all features using the fitted PCA model.
    print("Transforming all features using PCA...")
    pc_scores = pca.transform(features_scaled)

    # Binarize each of the 5 principal components using median thresholding.
    # Each column becomes 0/1 depending on whether it is below or above its median.
    print("Binarizing PCA components...")
    binary_labels = np.array([
        (pc_scores[:, i] > np.median(pc_scores[:, i])).astype(int)
        for i in range(n_bits)
    ]).T  # Shape: (n_samples, 5)

    # use the first log2(n_classes) bits to form integer labels.
    targets = [2 ** i for i in range(1, n_bits + 1)]  # e.g. creates [2, 4, 8, 16, 32, 64] for n_bits=6

    for target in targets:
        n_req = int(log2(target))
        # Combine the first n_req bits into an integer label.
        powers = 2 ** np.arange(n_req - 1, -1, -1)
        class_labels = np.dot(binary_labels[:, :n_req], powers)
        # (The dot product will yield integers in [0, target-1].)
        print(f"Created PCA classes for n_classes = {target}")

        # Save the labels along with the image names to CSV.
        df = pd.DataFrame({'image': image_names, 'pca_label': class_labels})
        output_csv = os.path.join(labels_dir, f"n_classes_{target}.csv")
        df.to_csv(output_csv, index=False)
        print(f"Saved {len(df)} PCA labels to {output_csv}")
        
        # Print class distribution statistics
        class_counts = df['pca_label'].value_counts().sort_index()
        print(f"\nClass distribution for {target} classes:")
        for class_idx, count in class_counts.items():
            print(f"Class {class_idx}: {count} images ({count/len(df)*100:.2f}%)")
        print(f"Average: {class_counts.mean():.1f} images/class")
        print(f"Min: {class_counts.min()} images (Class {class_counts.idxmin()})")
        print(f"Max: {class_counts.max()} images (Class {class_counts.idxmax()})")
        print()

if __name__ == '__main__':
    main()