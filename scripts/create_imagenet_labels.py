import os
import json
import argparse

def create_folder_labels(base_path: str) -> dict:
    """Create label mapping based on folder structure.
    
    Args:
        base_path: Path to ImageNet dataset root
        
    Returns:
        Dict mapping folder names to sequential labels (0 to num_folders-1)
    """
    folder_to_label = {}
    label_idx = 0
    
    # Create sequential labels
    for folder_name in sorted(os.listdir(base_path)):  # Sort for deterministic labels
        if not folder_name.startswith('n'):  # Skip non-class folders
            continue
            
        folder_path = os.path.join(base_path, folder_name)
        if not os.path.isdir(folder_path):
            continue
            
        folder_to_label[folder_name] = label_idx
        label_idx += 1
    
    return folder_to_label

def main():
    parser = argparse.ArgumentParser(description='Create ImageNet folder-based label mapping')
    parser.add_argument('--data_path', type=str, required=True,
                      help='Path to ImageNet dataset root')
    parser.add_argument('--output_path', type=str, default='datasets/obj_cls/imagenet',
                      help='Path to save label mapping file')
    args = parser.parse_args()
    
    # Create label mapping
    folder_to_label = create_folder_labels(args.data_path)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    
    # Save folder to label mapping
    label_file = os.path.join(args.output_path, 'folder_labels.json')
    with open(label_file, 'w') as f:
        json.dump(folder_to_label, f, indent=2)
    
    print(f"Found {len(folder_to_label)} class folders")
    print(f"Saved folder-to-label mapping to {label_file}")

if __name__ == '__main__':
    main() 