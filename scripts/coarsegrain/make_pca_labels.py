"""
Generate PCA-based coarse-grained labels for ImageNet.

Projects fc2 features onto principal components and applies global median splits
to create 2, 4, 8, ... classes (one threshold per PC).
"""
import os
import numpy as np
import pandas as pd

# Configuration
MODEL_NAME = "dino"
FEATURES_PATH = f"datasets/obj_cls/imagenet/features_{MODEL_NAME}.npz"
EIGENVECTORS_PATH = f"datasets/obj_cls/imagenet/eigenvectors_{MODEL_NAME}.npz"
N_PCS = 6  # Number of principal components (produces 2^N_PCS classes max)


def make_labels(scores):
    """Generate labels using global median threshold on each PC."""
    binary = (scores > np.median(scores, axis=0)).astype(int)
    for n_bits in range(1, scores.shape[1] + 1):
        powers = 2 ** np.arange(n_bits - 1, -1, -1)
        yield 2 ** n_bits, binary[:, :n_bits] @ powers


def main():
    print(f"Loading eigenvectors from {EIGENVECTORS_PATH}")
    pca = np.load(EIGENVECTORS_PATH)
    eigenvectors = pca['eigenvectors'][:, :N_PCS]
    mean = pca['mean']

    print(f"Loading features from {FEATURES_PATH}")
    data = np.load(FEATURES_PATH, allow_pickle=True)

    names = data['image_names']
    if names.size > 0 and isinstance(names[0], (bytes, np.bytes_)):
        names = [n.decode('utf-8') for n in names]
    names = [os.path.basename(str(n)) for n in names]

    features = data[f'{MODEL_NAME}_features'].reshape(len(names), -1)

    scores = (features - mean) @ eigenvectors

    labels_dir = f"pca_labels/pca_labels_{MODEL_NAME}"
    os.makedirs(labels_dir, exist_ok=True)

    print("Generating labels...")
    for n_classes, labels in make_labels(scores):
        df = pd.DataFrame({'image': names, 'pca_label': labels})
        df.to_csv(os.path.join(labels_dir, f"n_classes_{n_classes}.csv"), index=False)
        counts = df['pca_label'].value_counts()
        print(f"  {n_classes:2d} classes: min={counts.min():6d}, max={counts.max():6d}")


if __name__ == '__main__':
    main()
