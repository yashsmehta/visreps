"""Extract 1000 class centroids in PC1/PC2 space with majority-vote coarse labels.

For each ImageNet class (synset), samples N images, extracts AlexNet fc2 features,
projects onto PC1/PC2, and averages to get a single (x, y) centroid. Coarse labels
at each granularity (2, 4, 8, 16, 32, 64) are assigned by majority vote among
the sampled images' PCA labels.

Output: class_centroids_alexnet.npz with keys:
    - pc1, pc2: (1000,) arrays of centroid coordinates
    - synsets: (1000,) array of synset IDs
    - labels_{2,4,8,16,32,64}: (1000,) arrays of majority-vote coarse labels

Usage:
    python manuscript/figures/fig3/extract_class_centroids.py
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from collections import Counter
from dotenv import load_dotenv

load_dotenv()

IMAGENET_DIR = os.environ["IMAGENET_DATA_DIR"]
EIGENVECTORS_PATH = "datasets/obj_cls/imagenet/eigenvectors_alexnet.npz"
PCA_LABELS_DIR = "pca_labels/pca_labels_alexnet"
OUTPUT_PATH = "manuscript/figures/fig3/class_centroids_alexnet.npz"
IMAGES_PER_CLASS = 50
GRANULARITIES = [2, 4, 8, 16, 32, 64]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained AlexNet truncated to fc2
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(*list(model.classifier.children())[:6])
    model.eval().to(device)

    # Load eigenvectors for projection
    pca = np.load(EIGENVECTORS_PATH)
    eigenvectors = pca["eigenvectors"][:, :2]  # first 2 PCs
    mean = pca["mean"]

    # Load PCA labels (per-image) at each granularity
    label_dfs = {}
    for n in GRANULARITIES:
        df = pd.read_csv(os.path.join(PCA_LABELS_DIR, f"n_classes_{n}.csv"))
        label_dfs[n] = df.set_index("image")["pca_label"]

    # ImageNet dataset
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = ImageFolder(IMAGENET_DIR, transform=preprocess)

    # Group dataset indices by class
    class_to_indices = {}
    for idx, (_, class_idx) in enumerate(dataset.samples):
        class_to_indices.setdefault(class_idx, []).append(idx)

    # Map class_idx -> synset name
    idx_to_synset = {v: k for k, v in dataset.class_to_idx.items()}

    print(f"Found {len(class_to_indices)} classes, sampling {IMAGES_PER_CLASS} per class")

    # Sample indices and track which images belong to which class
    all_indices = []
    class_ranges = {}  # class_idx -> (start, end) in all_indices
    for class_idx in sorted(class_to_indices.keys()):
        indices = class_to_indices[class_idx]
        rng = np.random.RandomState(42)
        sampled = rng.choice(indices, size=min(IMAGES_PER_CLASS, len(indices)), replace=False)
        start = len(all_indices)
        all_indices.extend(sampled)
        class_ranges[class_idx] = (start, start + len(sampled))

    # Extract features in batches
    subset = Subset(dataset, all_indices)
    loader = DataLoader(subset, batch_size=256, num_workers=8, pin_memory=True)

    features_list = []
    with torch.no_grad():
        for images, _ in loader:
            features = model(images.to(device))
            features = F.normalize(features, p=2, dim=-1)
            features_list.append(features.cpu().numpy())

    all_features = np.concatenate(features_list, axis=0)
    print(f"Extracted features: {all_features.shape}")

    # Project onto PC1/PC2
    all_pc = (all_features - mean) @ eigenvectors

    # Compute centroids and majority-vote labels per class
    pc1_centroids = np.zeros(len(class_to_indices))
    pc2_centroids = np.zeros(len(class_to_indices))
    synsets = []
    majority_labels = {n: np.zeros(len(class_to_indices), dtype=int) for n in GRANULARITIES}

    for class_idx in sorted(class_to_indices.keys()):
        start, end = class_ranges[class_idx]
        synset = idx_to_synset[class_idx]
        synsets.append(synset)

        # Centroid = mean PC coordinates
        pc1_centroids[class_idx] = all_pc[start:end, 0].mean()
        pc2_centroids[class_idx] = all_pc[start:end, 1].mean()

        # Majority vote for coarse labels
        sampled_indices = all_indices[start:end]
        image_names = [os.path.basename(dataset.samples[i][0]) for i in sampled_indices]

        for n in GRANULARITIES:
            labels_for_images = []
            for name in image_names:
                if name in label_dfs[n].index:
                    labels_for_images.append(label_dfs[n][name])
            if labels_for_images:
                majority_labels[n][class_idx] = Counter(labels_for_images).most_common(1)[0][0]

    # Save
    save_dict = {
        "pc1": pc1_centroids,
        "pc2": pc2_centroids,
        "synsets": np.array(synsets),
    }
    for n in GRANULARITIES:
        save_dict[f"labels_{n}"] = majority_labels[n]

    np.savez_compressed(OUTPUT_PATH, **save_dict)
    print(f"Saved {len(synsets)} class centroids to {OUTPUT_PATH}")

    # Quick sanity check
    for n in GRANULARITIES:
        unique = len(np.unique(majority_labels[n]))
        print(f"  {n}-way: {unique} unique labels")


if __name__ == "__main__":
    main()
