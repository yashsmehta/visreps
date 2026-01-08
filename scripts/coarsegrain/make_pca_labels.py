import os
import numpy as np
import pandas as pd

# Configuration
MODEL_NAME = "alexnet"
FEATURES_PATH = f"datasets/obj_cls/imagenet/features_{MODEL_NAME}.npz"
EIGENVECTORS_PATH = f"datasets/obj_cls/imagenet/eigenvectors_{MODEL_NAME}.npz"
N_LEVELS = 6
HIERARCHICAL = True  # True: balanced splits, False: global median thresholds


def make_labels_hierarchical(scores):
    """Recursive bisection: each group split by median of next PC."""
    n_samples, n_levels = scores.shape
    labels = np.zeros(n_samples, dtype=int)

    for level in range(n_levels):
        pc_scores = scores[:, level]
        new_labels = np.zeros(n_samples, dtype=int)

        for group_id in range(2 ** level):
            idx = np.where(labels == group_id)[0]
            if len(idx) == 0:
                continue
            sorted_order = np.argsort(pc_scores[idx])
            half = len(idx) // 2
            new_labels[idx[sorted_order[:half]]] = group_id * 2
            new_labels[idx[sorted_order[half:]]] = group_id * 2 + 1

        labels = new_labels
        yield 2 ** (level + 1), labels.copy()


def make_labels_global(scores):
    """Global median threshold on each PC independently."""
    binary = (scores > np.median(scores, axis=0)).astype(int)
    for n_bits in range(1, scores.shape[1] + 1):
        powers = 2 ** np.arange(n_bits - 1, -1, -1)
        yield 2 ** n_bits, binary[:, :n_bits] @ powers


def main():
    print(f"Loading PCA model from {EIGENVECTORS_PATH}")
    pca = np.load(EIGENVECTORS_PATH)
    eigenvectors, mean = pca['eigenvectors'][:, :N_LEVELS], pca['mean']

    print(f"Loading features from {FEATURES_PATH}")
    data = np.load(FEATURES_PATH, allow_pickle=True)
    names = data['image_names']
    if names.size > 0 and isinstance(names[0], (bytes, np.bytes_)):
        names = [n.decode('utf-8') for n in names]
    names = [os.path.basename(str(n)) for n in names]

    for key in ['fc2', 'clip_features', 'features', 'dreamsim_features']:
        if key in data:
            features = data[key].reshape(len(names), -1)
            break

    scores = (features - mean) @ eigenvectors

    method = "hierarchical" if HIERARCHICAL else "global"
    labels_dir = os.path.join("pca_labels", f"pca_labels_{MODEL_NAME}_{method}")
    os.makedirs(labels_dir, exist_ok=True)

    print(f"Generating labels ({method} method)...")
    label_gen = make_labels_hierarchical(scores) if HIERARCHICAL else make_labels_global(scores)

    for n_classes, labels in label_gen:
        df = pd.DataFrame({'image': names, 'pca_label': labels})
        df.to_csv(os.path.join(labels_dir, f"n_classes_{n_classes}.csv"), index=False)
        counts = df['pca_label'].value_counts()
        print(f"  {n_classes:2d} classes: min={counts.min():6d}, max={counts.max():6d}")


if __name__ == '__main__':
    main()
