import nltk
from nltk.corpus import wordnet as wn
import os
import sys

# Add project root to path so we can import from visreps
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Mock config and utils if needed, or just use the dataset directly
import visreps.utils as utils
from visreps.dataloaders.obj_cls import ImageNetDataset

def setup():
    """Download necessary NLTK data."""
    try:
        wn.ensure_loaded()
    except LookupError:
        nltk.download('wordnet')
        nltk.download('omw-1.4')

def print_hierarchy(synset, depth=0, max_depth=3, max_children=5):
    """
    Recursively print the hyponym (child) hierarchy of a synset.
    
    Args:
        synset: The NLTK Synset object.
        depth: Current depth in the tree.
        max_depth: Maximum depth to traverse.
        max_children: Maximum number of children to display per node.
    """
    indent = "  " * depth
    print(f"{indent}- {synset.name()} ({synset.definition().split(';')[0]})")
    
    if depth >= max_depth:
        return

    hyponyms = synset.hyponyms()
    
    # Sort by name for consistent output
    hyponyms.sort(key=lambda s: s.name())
    
    for i, child in enumerate(hyponyms):
        if i >= max_children:
            print(f"{indent}  ... ({len(hyponyms) - max_children} more)")
            break
        print_hierarchy(child, depth + 1, max_depth, max_children)

def print_ancestry(synset):
    """
    Print the ancestry (hypernym paths) of a synset up to the root.
    """
    print(f"Ancestry for: {synset.name()} ({synset.definition().split(';')[0]})")
    
    # hypernym_paths() returns a list of lists (multiple paths to root possible)
    paths = synset.hypernym_paths()
    
    for i, path in enumerate(paths):
        print(f"\nPath {i+1}:")
        for depth, node in enumerate(path):
            indent = "  " * depth
            print(f"{indent}- {node.name()} ({node.definition().split(';')[0]})")
            
def main():
    setup()
    
    # 1. Visualization of WordNet hierarchy (children)
    print("=== 1. Visualization of WordNet Children (General) ===\n")
    
    root_term = 'carnivore.n.01'
    print(f"Visualizing subset starting from: {root_term}\n")
    try:
        root_synset = wn.synset(root_term)
        print_hierarchy(root_synset, max_depth=2, max_children=4)
    except Exception as e:
        print(f"Error accessing synset '{root_term}': {e}")

    print("\n" + "="*50 + "\n")

    # 2. Link ImageNet Class to Hierarchy
    print("=== 2. Link ImageNet Class to Hierarchy ===\n")
    
    try:
        # Instantiate dataset to load label mapping
        ds = ImageNetDataset(utils.get_env_var("IMAGENET_DATA_DIR"), split="train")
        
        # Examples: 0 (tench), 250 (Siberian husky), 999 (toilet tissue)
        for idx in [0, 250, 999]:
            print(f"\n--- Analyzing ImageNet Class Index: {idx} ---")
            if synset := ds.get_wordnet_synset(idx):
                print_ancestry(synset)
            else:
                print("Could not retrieve synset.")
                
    except Exception as e:
        print(f"Skipping ImageNet integration test: {e}")

if __name__ == "__main__":
    main()
