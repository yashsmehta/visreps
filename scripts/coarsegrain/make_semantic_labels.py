"""Expand a hand-made class-to-group mapping into per-image coarse labels.

Reads ``pca_labels/pca_labels_semantic/class_to_group.csv`` (one row per
ImageNet class: class_idx, wnid, class_name, group_id, group_name) and writes
``pca_labels/pca_labels_semantic/n_classes_{K}.csv`` in the same
``image,pca_label`` format as the PCA and WordNet label files.

The image list is copied from an existing label file so every training image
gets a label. Each image's wnid is the prefix of its filename (``n03729826_8440.JPEG``).
"""
import os
import pandas as pd

MAPPING_PATH = "pca_labels/pca_labels_semantic/class_to_group.csv"
IMAGE_LIST_SOURCE = "pca_labels/pca_labels_alexnet/n_classes_8.csv"
OUT_DIR = "pca_labels/pca_labels_semantic"


def main():
    mapping = pd.read_csv(MAPPING_PATH)
    n_classes = mapping["group_id"].nunique()
    assert sorted(mapping["group_id"].unique()) == list(range(n_classes))
    wnid_to_group = dict(zip(mapping["wnid"], mapping["group_id"]))

    images = pd.read_csv(IMAGE_LIST_SOURCE, usecols=["image"])
    wnids = images["image"].str.split("_").str[0]
    assert wnids.isin(wnid_to_group).all(), "image with unknown wnid"

    out = pd.DataFrame({"image": images["image"], "pca_label": wnids.map(wnid_to_group)})
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"n_classes_{n_classes}.csv")
    out.to_csv(out_path, index=False)

    print(f"Wrote {len(out):,} image labels to {out_path}")
    print("Images per group:")
    counts = out["pca_label"].value_counts().sort_index()
    names = mapping.drop_duplicates("group_id").set_index("group_id")["group_name"]
    for g, c in counts.items():
        print(f"  {g} {names[g]:<28} {c:>8,}")


if __name__ == "__main__":
    main()
