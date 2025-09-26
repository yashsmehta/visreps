import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

FEATURES_PATH = "/data/apassi1/layer_activations/ImageNet/layer11.npz"
LABELS_FOLDER = "pca_labels_scattering11"  # Directory name for saving labels
N_BITS = 6  # Number of PCA components

def main():
    # Load data
    print(f"\nProcessing features from: {FEATURES_PATH}")
    data_dict = np.load(FEATURES_PATH, allow_pickle=True)
    
    # Process image names
    image_names = data_dict['image_names']
    if image_names.size > 0 and isinstance(image_names[0], (bytes, np.bytes_)):
        image_names = [name.decode('utf-8') for name in image_names]
    image_names = [os.path.basename(str(name)) for name in image_names]

    # Detect feature type
    feature_keys = {'fc2': True, 'clip_features': True, 'features': True}
    for key in feature_keys:
        if key in data_dict:
            feature_array = data_dict[key]
            break
    else:
        raise ValueError(f"No valid feature key found. Available: {list(data_dict.keys())}")

    # Set output directory
    labels_dir = os.path.join("datasets", "obj_cls", "imagenet", LABELS_FOLDER)
    os.makedirs(labels_dir, exist_ok=True)
    print(f"Saving PCA labels to: {labels_dir}")
    
    # Align features and image names
    n_samples = min(feature_array.shape[0], len(image_names))
    feature_array = feature_array[:n_samples]
    image_names = image_names[:n_samples]

    # Reshape to 2D if needed
    features_2d = feature_array.reshape(n_samples, -1) if feature_array.ndim != 2 else feature_array

    # Fit PCA
    features_scaled = StandardScaler().fit_transform(features_2d)
    n_fit = min(110000, n_samples)
    np.random.seed(42)
    fit_indices = np.random.choice(n_samples, n_fit, replace=False)
    pca = PCA(n_components=N_BITS)
    pca.fit(features_scaled[fit_indices])

    print(f"\nFitted PCA with {N_BITS} components on {n_fit} samples")
    print(f"Total variance explained by first {N_BITS} PCs: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")
    print(f"Dataset size: {n_samples} images")

    # Transform and binarize
    pc_scores = pca.transform(features_scaled)
    binary_labels = (pc_scores > np.median(pc_scores, axis=0)).astype(int)

    # Generate and save class labels for different granularities
    print("\nGenerating class labels:")
    for n_bits_used in range(1, N_BITS + 1):
        n_classes = 2 ** n_bits_used
        powers = 2 ** np.arange(n_bits_used - 1, -1, -1)
        class_labels = np.dot(binary_labels[:, :n_bits_used], powers)

        # Save CSV
        df = pd.DataFrame({'image': image_names, 'pca_label': class_labels})
        output_csv = os.path.join(labels_dir, f"n_classes_{n_classes}.csv")
        df.to_csv(output_csv, index=False)

        # Print summary with min/max per class
        class_counts = df['pca_label'].value_counts()
        print(f"  n_classes={n_classes:2d}: min={class_counts.min():4d}, max={class_counts.max():4d} per class")

if __name__ == '__main__':
    main()