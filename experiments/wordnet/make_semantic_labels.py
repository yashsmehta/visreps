"""
Create semantic labels for ImageNet based on semantic super-categories.
Groups 64 Level-6 WordNet synsets into semantically meaningful categories.
"""
import os
import sys
import pandas as pd
from collections import Counter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import nltk
from nltk.corpus import wordnet as wn

import visreps.utils as utils
from visreps.dataloaders.obj_cls import ImageNetDataset

SUPER_CATEGORIES = {
    'Animals': [
        'animal.n.01'
    ],
    'Natural World': [
        # Landscapes + Biological growth
        'plant.n.02', 'plant_organ.n.01', 'fungus.n.01', 
        'alp.n.01', 'cliff.n.01', 'reef.n.01', 'dune.n.01', 
        'geyser.n.01', 'lakeside.n.01', 'lunar_crater.n.01', 
        'promontory.n.01', 'bar.n.08', 'seashore.n.01', 
        'valley.n.01', 'volcano.n.02'
    ],
    'Food & Produce': [
        'vegetable.n.01', 'edible_fruit.n.01', 'starches.n.01'
    ],
    'Structures & Architecture': [
        'building.n.01', 'establishment.n.04', 'obstruction.n.01', 
        'protective_covering.n.01', 'top.n.09', 'memorial.n.03', 
        'tower.n.01', 'supporting_structure.n.01', 'housing.n.01', 
        'column.n.06', 'bridge.n.01', 'defensive_structure.n.01', 
        'coil.n.01', 'colonnade.n.01', 'landing.n.02', 'fountain.n.01', 
        'house_of_cards.n.02', 'building_complex.n.01', 'stadium.n.01', 
        'shelter.n.01', 'pool.n.01', 'workplace.n.01', 'arch.n.04'
    ],
    'Domestic & Apparel': [
        # Items for the body or the room
        'clothing.n.01', 'footwear.n.02', 'cloth_covering.n.01', 'towel.n.01', 
        'bib.n.01', 'dishrag.n.01', 'handkerchief.n.01', 'mask.n.01', 
        'furnishing.n.02', 'floor_cover.n.01', 'toiletry.n.01', 'powder.n.03'
    ],
    'Vehicles & Transport': [
        'conveyance.n.03'
    ],
    'Tools & Electronics': [
        'device.n.01', 'equipment.n.01', 'implement.n.01', 
        'system.n.01', 'memory.n.04', 'medium.n.01'
    ],
    'General Objects': [
        # Miscellaneous goods and building materials
        'container.n.01', 'consumer_goods.n.01', 'product.n.02', 
        'brick.n.01', 'coating.n.01', 'screen.n.04'
    ]
}
# Build reverse mapping: synset -> super-category
SYNSET_TO_SUPER = {}
for super_cat, synsets in SUPER_CATEGORIES.items():
    for syn in synsets:
        SYNSET_TO_SUPER[syn] = super_cat

# Category order derived from SUPER_CATEGORIES (Python 3.7+ dicts preserve insertion order)
CATEGORY_ORDER = list(SUPER_CATEGORIES.keys())

# Output to the same directory as this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "semantic_categories.csv")


def setup_wordnet():
    """Ensure WordNet is loaded."""
    try:
        wn.ensure_loaded()
    except LookupError:
        nltk.download('wordnet')
        nltk.download('omw-1.4')


def get_level6_synset(synset):
    """Get the Level 6 synset from the hypernym path."""
    if synset is None:
        return None
    
    paths = synset.hypernym_paths()
    if not paths:
        return None
    
    path = min(paths, key=len)
    
    if len(path) > 6:
        return path[6]
    elif len(path) > 0:
        return path[-1]
    return None


def main():
    setup_wordnet()
    
    print("Loading ImageNet dataset...")
    base_path = utils.get_env_var("IMAGENET_DATA_DIR")
    ds = ImageNetDataset(base_path, split="all")
    print(f"Loaded {len(ds.samples)} images\n")
    
    # Create category to label mapping
    category_to_label = {cat: i for i, cat in enumerate(CATEGORY_ORDER)}
    
    # Classify each ImageNet class
    print(f"Classifying 1000 ImageNet classes into {len(CATEGORY_ORDER)} super-categories...")
    class_to_category = {}
    category_counts = Counter()
    unmapped_synsets = set()
    
    for class_idx in range(1000):
        synset = ds.get_wordnet_synset(class_idx)
        level6_synset = get_level6_synset(synset)

        if level6_synset is None:
            raise ValueError(f"Class {class_idx} ({synset}) has no Level 6 synset")

        level6_name = level6_synset.name()
        if level6_name not in SYNSET_TO_SUPER:
            unmapped_synsets.add(level6_name)
        else:
            super_cat = SYNSET_TO_SUPER[level6_name]
            class_to_category[class_idx] = super_cat
            category_counts[super_cat] += 1

    if unmapped_synsets:
        print(f"\nERROR: {len(unmapped_synsets)} unmapped Level 6 synsets. Add these to SUPER_CATEGORIES:")
        for syn in sorted(unmapped_synsets):
            print(f"  - {syn}")
        sys.exit(1)
    
    # Print distribution
    print("\n" + "=" * 60)
    print(f"FINAL CATEGORY DISTRIBUTION ({len(CATEGORY_ORDER)} Super-Categories)")
    print("=" * 60)
    total = 0
    for cat in CATEGORY_ORDER:
        label = category_to_label[cat]
        count = category_counts[cat]
        total += count
        print(f"  {label}: {cat:<20} {count:4} classes")
    print(f"\nTotal: {total} classes")
    
    # Build image → label mapping
    print(f"\nAssigning labels to {len(ds.samples)} images...")
    records = []
    for img_path, class_idx, img_id in ds.samples:
        category = class_to_category[class_idx]
        label = category_to_label[category]
        records.append({"image": img_id, "pca_label": label})
    
    # Save CSV
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to {OUTPUT_FILE}")
    
    # Save category mapping
    mapping_file = OUTPUT_FILE.replace(".csv", "_mapping.txt")
    with open(mapping_file, "w") as f:
        f.write(f"{len(CATEGORY_ORDER)} Super-Categories for ImageNet\n")
        f.write("=" * 60 + "\n\n")
        for cat in CATEGORY_ORDER:
            label = category_to_label[cat]
            count = category_counts[cat]
            synsets = SUPER_CATEGORIES[cat]
            f.write(f"{label}: {cat} ({count} classes)\n")
            f.write(f"   Level 6 synsets: {', '.join(synsets)}\n\n")
    print(f"Saved mapping to {mapping_file}")
    
    # Print category names for visualization
    print("\n" + "=" * 60)
    print("CATEGORY_NAMES for 2d_visualization.py:")
    print("=" * 60)
    print(f"CATEGORY_NAMES = {CATEGORY_ORDER}")


if __name__ == "__main__":
    main()
