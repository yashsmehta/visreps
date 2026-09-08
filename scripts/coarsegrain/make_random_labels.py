"""Random class-level 8-way labels: the control for the handcrafted semantic labels.

Each of the 1,000 ImageNet classes (wnids) is assigned to one of eight groups
uniformly at random, balanced at 125 wnids per group, with a fixed seed. All
images of a class share its group, exactly as for the handcrafted labels in
``pca_labels/pca_labels_semantic/``; only the semantic coherence of the groups
is removed. Writes ``pca_labels/pca_labels_random/n_classes_8.csv`` in the
same format as the other label folders.
"""
import argparse
import os
import numpy as np
import pandas as pd

IMAGE_LIST_SOURCE = "pca_labels/pca_labels_semantic/n_classes_8.csv"
OUT_DIR = "pca_labels/pca_labels_random"
N_CLASSES = 8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    images = pd.read_csv(IMAGE_LIST_SOURCE, usecols=["image"])
    wnids = images["image"].str.split("_").str[0]
    unique = np.array(sorted(wnids.unique()))
    assert len(unique) % N_CLASSES == 0, f"{len(unique)} wnids not divisible by {N_CLASSES}"

    rng = np.random.default_rng(args.seed)
    permuted = rng.permutation(unique)
    wnid_to_label = {w: i % N_CLASSES for i, w in enumerate(permuted)}

    out = pd.DataFrame({"image": images["image"], "pca_label": wnids.map(wnid_to_label)})
    os.makedirs(OUT_DIR, exist_ok=True)
    out.to_csv(os.path.join(OUT_DIR, f"n_classes_{N_CLASSES}.csv"), index=False)
    pd.DataFrame({"wnid": unique, "label": [wnid_to_label[w] for w in unique]}).to_csv(
        os.path.join(OUT_DIR, "class_assignments.csv"), index=False)

    print(f"seed={args.seed}  {len(unique)} wnids -> {N_CLASSES} groups")
    print("wnids per group:", pd.Series(wnid_to_label).value_counts().sort_index().tolist())
    print("images per group:", out["pca_label"].value_counts().sort_index().tolist())


if __name__ == "__main__":
    main()
