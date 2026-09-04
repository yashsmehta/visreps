"""Build per-image coarse labels from hand-assigned binary semantic dimensions.

``pca_labels/pca_labels_semantic/class_dimensions.csv`` holds one row per
ImageNet class with six binary columns (natural, handheld, indoor, self_moving,
soft, elongated; definitions in ``dimension_definitions.md``). Pick k of them and
every class gets label = binary number formed by those bits, giving 2^k classes.

Usage (from project root):
    python scripts/coarsegrain/make_semantic_labels.py --dims natural handheld indoor

Writes ``pca_labels/pca_labels_semantic/n_classes_{2^k}.csv`` in the same
``image,pca_label`` format as the PCA label files. The image list is copied from
an existing label file; each image's wnid is the prefix of its filename.
"""
import argparse
import os
import pandas as pd

DIMS_PATH = "pca_labels/pca_labels_semantic/class_dimensions.csv"
IMAGE_LIST_SOURCE = "pca_labels/pca_labels_alexnet/n_classes_8.csv"
OUT_DIR = "pca_labels/pca_labels_semantic"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dims", nargs="+", required=True, help="dimension columns, most significant bit first")
    args = parser.parse_args()

    dims = pd.read_csv(DIMS_PATH)
    missing = [d for d in args.dims if d not in dims.columns]
    assert not missing, f"unknown dimensions: {missing}"

    label = sum(dims[d] * 2 ** (len(args.dims) - 1 - i) for i, d in enumerate(args.dims))
    wnid_to_label = dict(zip(dims["wnid"], label))
    n_classes = 2 ** len(args.dims)

    images = pd.read_csv(IMAGE_LIST_SOURCE, usecols=["image"])
    wnids = images["image"].str.split("_").str[0]
    assert wnids.isin(wnid_to_label).all(), "image with unknown wnid"

    out = pd.DataFrame({"image": images["image"], "pca_label": wnids.map(wnid_to_label)})
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"n_classes_{n_classes}.csv")
    out.to_csv(out_path, index=False)

    print(f"Dimensions (MSB first): {args.dims}")
    print(f"Wrote {len(out):,} image labels to {out_path}")
    print("Classes and images per label:")
    per_class = pd.Series(wnid_to_label).value_counts().sort_index()
    per_image = out["pca_label"].value_counts().sort_index()
    for lbl in range(n_classes):
        bits = format(lbl, f"0{len(args.dims)}b")
        print(f"  {lbl} ({bits})  classes={per_class.get(lbl, 0):4d}  images={per_image.get(lbl, 0):>8,}")


if __name__ == "__main__":
    main()
