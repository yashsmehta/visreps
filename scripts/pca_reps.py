import os
import argparse
import numpy as np
import pandas as pd
from math import log2
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='imagenet',
                        choices=['tiny-imagenet', 'imagenet'],
                        help='Dataset to process features from (default: imagenet)')
    # Add argument to specify which feature file to process
    parser.add_argument('--pretrained_source', type=str, required=True, choices=['none', 'imagenet1k'],
                        help='Specify which feature set to process: none or imagenet1k')
    args = parser.parse_args()

    # Use first n_bits PCA components (i.e. n_bits=6)
    n_bits = 6
    # Use the argument for pretrained dataset source
    pretrained_dataset = args.pretrained_source 

    # Construct paths based on the argument
    base_dir = os.path.join("datasets", "obj_cls", args.dataset)
    features_path = os.path.join(base_dir, f"features_pretrained_{pretrained_dataset}.npz")
    labels_dir = os.path.join(base_dir, f"pca_labels_{pretrained_dataset}")
    
    print(f"\nProcessing features from: {features_path}")
    print(f"Saving PCA labels to: {labels_dir}")

    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found at {features_path}. Please run extract_reps.py first.")
    os.makedirs(labels_dir, exist_ok=True)
    
    print(f"Loading features from {features_path}")
    try:
        data_dict = np.load(features_path, allow_pickle=True)
    except Exception as e:
        raise RuntimeError(f"Error loading features file {features_path}: {e}")

    # Process image names.
    if 'image_names' not in data_dict:
        raise ValueError(f"'image_names' key not found in {features_path}")
    image_names = data_dict['image_names']
    print(f"Loaded {len(image_names)} image names from {features_path}")
    if image_names.size > 0 and isinstance(image_names[0], (bytes, np.bytes_)):
        try:
            image_names = [name.decode('utf-8') for name in image_names]
        except UnicodeDecodeError:
            raise ValueError(f"Failed to decode image names in {features_path}")
    image_names = [os.path.basename(str(name)) for name in image_names]

    # Verify and reshape features if necessary.
    if 'fc2' not in data_dict:
        raise ValueError(f"fc2 features not found in {features_path}")
    feature_array = data_dict['fc2']
    n_samples = feature_array.shape[0]
    
    if n_samples != len(image_names):
         print(f"Warning: Number of samples in features ({n_samples}) does not match number of image names ({len(image_names)}) in {features_path}. Using min count.")
         min_count = min(n_samples, len(image_names))
         feature_array = feature_array[:min_count]
         image_names = image_names[:min_count]
         n_samples = min_count

    if n_samples == 0:
        print("No samples found in features file. Exiting.")
        return

    if feature_array.ndim != 2:
        try:
            features_2d = feature_array.reshape(n_samples, -1)
            print(f"Reshaped features from {feature_array.shape} to {features_2d.shape}")
        except ValueError as e:
            raise ValueError(f"Error reshaping features from {features_path}: {e}. Original shape: {feature_array.shape}")
    else:
        features_2d = feature_array

    # Standardize features.
    print("Standardizing features...")
    features_scaled = StandardScaler().fit_transform(features_2d)

    # Fit PCA on 110K random samples (or all if fewer than 110K).
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
    # print(f"Singular values:", pca.singular_values_)

    # Transform all features using the fitted PCA model.
    print("Transforming all features using PCA...")
    pc_scores = pca.transform(features_scaled)

    # Binarize each of the n_bits principal components using median thresholding.
    print("Binarizing PCA components...")
    binary_labels = np.array([
        (pc_scores[:, i] > np.median(pc_scores[:, i])).astype(int)
        for i in range(n_bits)
    ]).T  

    # use the first log2(n_classes) bits to form integer labels.
    targets = [2 ** i for i in range(1, n_bits + 1)]  # [2, 4, 8, 16, 32, 64]

    for target in targets:
        n_req = int(log2(target))
        # Combine the first n_req bits into an integer label.
        powers = 2 ** np.arange(n_req - 1, -1, -1)
        class_labels = np.dot(binary_labels[:, :n_req], powers)
        
        print(f"Created PCA classes for n_classes = {target}")
        
        print(f"DEBUG (n_classes={target}): len(image_names) = {len(image_names)}, len(class_labels) = {len(class_labels)}")

        # Save the labels along with the image names to CSV.
        df = pd.DataFrame({'image': image_names, 'pca_label': class_labels})
        output_csv = os.path.join(labels_dir, f"n_classes_{target}.csv")
        # Use pyarrow engine for potentially faster CSV writing
        df.to_csv(output_csv, index=False) 
        print(f"Saved {len(df)} PCA labels to {output_csv}")
        
        # Print class distribution statistics
        class_counts = df['pca_label'].value_counts().sort_index()
        print(f"\nClass distribution for {target} classes:")
        if not class_counts.empty:
            for class_idx, count in class_counts.items():
                print(f"Class {class_idx}: {count} images ({count/len(df)*100:.2f}%)")
            print(f"Average: {class_counts.mean():.1f} images/class")
            print(f"Min: {class_counts.min()} images (Class {class_counts.idxmin()})")
            print(f"Max: {class_counts.max()} images (Class {class_counts.idxmax()})")
        else:
            print("No classes found to compute distribution.")
        print()

if __name__ == '__main__':
    main()