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
                        choices=['tiny-imagenet', 'imagenet', 'imagenet-mini-50'],
                        help='Dataset to process features from (default: imagenet)')
    parser.add_argument('--features_filename', type=str, required=True,
                        help='Name of the features file to process (e.g., features_alexnet.npz)')
    args = parser.parse_args()

    # Setup paths
    base_dir = os.path.join("datasets", "obj_cls", args.dataset)
    features_path = os.path.join(base_dir, args.features_filename)
    
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Features file not found at {features_path}. Please run extract_reps.py first.")
    
    print(f"\nProcessing features from: {features_path}")
    
    # Load data
    data_dict = np.load(features_path, allow_pickle=True)
    
    # Process image names
    if 'image_names' not in data_dict:
        raise ValueError(f"'image_names' key not found in {features_path}")
    image_names = data_dict['image_names']
    if image_names.size > 0 and isinstance(image_names[0], (bytes, np.bytes_)):
        image_names = [name.decode('utf-8') for name in image_names]
    image_names = [os.path.basename(str(name)) for name in image_names]
    print(f"Loaded {len(image_names)} image names")

    # Detect feature type and load features
    if 'fc2' in data_dict:
        feature_array = data_dict['fc2']
        feature_type = "AlexNet fc2"
        pretrained_source = args.features_filename.replace('features_', '').replace('.npz', '')
        labels_dir = os.path.join(base_dir, f"pca_labels_{pretrained_source}")
    elif 'clip_features' in data_dict:
        feature_array = data_dict['clip_features']
        feature_type = "CLIP"
        labels_dir = os.path.join(base_dir, "pca_labels_clip")
    else:
        available_keys = list(data_dict.keys())
        raise ValueError(f"Neither 'fc2' nor 'clip_features' found in {features_path}. Available keys: {available_keys}")
    
    print(f"Using {feature_type} features for PCA analysis")
    os.makedirs(labels_dir, exist_ok=True)
    print(f"Saving PCA labels to: {labels_dir}")
    
    # Align features and image names
    n_samples = feature_array.shape[0]
    if n_samples != len(image_names):
        min_count = min(n_samples, len(image_names))
        feature_array = feature_array[:min_count]
        image_names = image_names[:min_count]
        n_samples = min_count
        print(f"Warning: Aligned to {n_samples} samples")

    if n_samples == 0:
        print("No samples found in features file. Exiting.")
        return

    # Reshape features if needed
    if feature_array.ndim != 2:
        features_2d = feature_array.reshape(n_samples, -1)
        print(f"Reshaped features from {feature_array.shape} to {features_2d.shape}")
    else:
        features_2d = feature_array

    # PCA processing
    n_bits = 6
    features_scaled = StandardScaler().fit_transform(features_2d)
    
    # Fit PCA on subset of data
    n_fit = min(110000, n_samples)
    np.random.seed(42)
    fit_indices = np.random.choice(n_samples, n_fit, replace=False)
    pca = PCA(n_components=n_bits)
    pca.fit(features_scaled[fit_indices])
    print(f"Fitted PCA with {n_bits} components on {n_fit} samples")

    # Print PCA statistics
    print("\nPCA Statistics:")
    for i, var_ratio in enumerate(pca.explained_variance_ratio_):
        print(f"PC{i+1}: {var_ratio:.4f} ({var_ratio*100:.2f}%)")
    print(f"Total variance explained: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")

    # Transform and binarize
    pc_scores = pca.transform(features_scaled)
    binary_labels = np.array([
        (pc_scores[:, i] > np.median(pc_scores[:, i])).astype(int)
        for i in range(n_bits)
    ]).T

    # Generate class labels for different numbers of classes
    targets = [2 ** i for i in range(1, n_bits + 1)]  # [2, 4, 8, 16, 32, 64]

    for target in targets:
        n_req = int(log2(target))
        powers = 2 ** np.arange(n_req - 1, -1, -1)
        class_labels = np.dot(binary_labels[:, :n_req], powers)
        
        # Save CSV
        df = pd.DataFrame({'image': image_names, 'pca_label': class_labels})
        output_csv = os.path.join(labels_dir, f"n_classes_{target}.csv")
        df.to_csv(output_csv, index=False)
        print(f"Saved {len(df)} PCA labels to {output_csv}")
        
        # Print distribution
        class_counts = df['pca_label'].value_counts().sort_index()
        print(f"Class distribution for {target} classes:")
        if not class_counts.empty:
            for class_idx, count in class_counts.items():
                print(f"  Class {class_idx}: {count} images ({count/len(df)*100:.2f}%)")
            print(f"  Average: {class_counts.mean():.1f} images/class")
        print()

if __name__ == '__main__':
    main()