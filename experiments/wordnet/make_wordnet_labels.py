import os
import sys
import pandas as pd
from collections import defaultdict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import visreps.utils as utils
from visreps.dataloaders.obj_cls import ImageNetDataset

LABELS_FOLDER = "wordnet"
MIN_DEPTH, MAX_DEPTH = 1, 7


def get_class_to_ancestor_at_depth(ds, depth):
    """Map each ImageNet class (0-999) to its ancestor synset name at given depth."""
    class_to_ancestor = {}
    
    for class_idx in range(1000):
        synset = ds.get_wordnet_synset(class_idx)
        if not synset:
            continue
            
        # Get longest hypernym path (most specific route to root)
        paths = synset.hypernym_paths()
        path = max(paths, key=len)
        
        # Get ancestor at depth (or leaf if path is shorter)
        ancestor_idx = min(depth, len(path) - 1)
        class_to_ancestor[class_idx] = path[ancestor_idx].name()
    
    return class_to_ancestor


def main():
    print("Loading ImageNet dataset...")
    base_path = utils.get_env_var("IMAGENET_DATA_DIR")
    ds = ImageNetDataset(base_path, split="all")
    print(f"Loaded {len(ds.samples)} images\n")

    # Output directory
    labels_dir = os.path.join("pca_labels", LABELS_FOLDER)
    os.makedirs(labels_dir, exist_ok=True)
    print(f"Saving WordNet labels to: {labels_dir}\n")

    print("Depth | # Classes | Output File")
    print("-" * 50)

    for depth in range(MIN_DEPTH, MAX_DEPTH + 1):
        # Get class → ancestor mapping
        class_to_ancestor = get_class_to_ancestor_at_depth(ds, depth)
        
        # Get unique ancestors and create label mapping
        unique_ancestors = sorted(set(class_to_ancestor.values()))
        ancestor_to_label = {a: i for i, a in enumerate(unique_ancestors)}
        n_classes = len(unique_ancestors)

        # Build image → label mapping
        records = []
        for img_path, class_idx, img_id in ds.samples:
            ancestor = class_to_ancestor.get(class_idx)
            if ancestor:
                label = ancestor_to_label[ancestor]
                records.append({"image": img_id, "pca_label": label})

        # Save CSV
        df = pd.DataFrame(records)
        output_path = os.path.join(labels_dir, f"n_classes_{n_classes}.csv")
        df.to_csv(output_path, index=False)
        
        print(f"{depth:5d} | {n_classes:9d} | {output_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()

