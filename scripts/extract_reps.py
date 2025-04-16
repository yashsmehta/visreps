import os
import argparse
import torch
import numpy as np
from omegaconf import OmegaConf
from visreps.models import standard_cnn
from visreps.models.utils import configure_feature_extractor
from visreps.dataloaders.obj_cls import get_obj_cls_loader

def get_model_and_layers(dataset, pretrained_dataset="imagenet1k"):
    """Initializes the model and defines the layers to extract features from.
    """
    num_classes = 200 if dataset == "tiny-imagenet" else 1000
    model = standard_cnn.AlexNet(pretrained_dataset=pretrained_dataset, num_classes=num_classes)
    
    layers = [
        'fc2'
    ]
    return model, layers

def extract_features(model, loader, device, layer_names):
    """Extracts activations using the feature extractor."""
    acts_dict = {name: [] for name in layer_names}
    all_image_names, all_labels = [], []
    
    # Need to handle potential empty dataset
    if len(loader.dataset) == 0:
        print("Warning: Dataset is empty, skipping feature extraction.")
        # Return empty dict and lists, ensuring keys exist in acts_dict
        for name in layer_names:
            acts_dict[name] = np.array([]) # Empty numpy array
        return acts_dict, [], []
        
    total_batches = len(loader)
    if total_batches == 0:
        print("Warning: DataLoader has 0 batches, skipping feature extraction.")
        for name in layer_names:
            acts_dict[name] = np.array([])
        return acts_dict, [], []

    for batch_idx, (images, labels) in enumerate(loader):
        print(f"Processing batch {batch_idx+1}/{total_batches}", end='\r')
        images = images.to(device)

        # Get image names from dataset - Adjusted for potentially empty batches or lists
        start = batch_idx * loader.batch_size
        end = start + images.size(0)
        # Check if dataset.samples exists and is indexable
        if hasattr(loader.dataset, 'samples') and len(loader.dataset.samples) > start:
            # Ensure end index doesn't exceed length
            current_end = min(end, len(loader.dataset.samples))
            names = [os.path.basename(loader.dataset.samples[i][2]) for i in range(start, current_end)]
        else:
            names = [] # Handle cases where samples aren't available or batch is beyond samples
        
        with torch.no_grad():
            features = model(images)  # Returns dict of layer activations

        # Save each layer's activation
        for name in layer_names:
            # Ensure features dict contains the key
            if name in features:
                acts_dict[name].append(features[name].cpu().numpy())
            else:
                print(f"Warning: Layer {name} not found in model output for batch {batch_idx+1}")
        
        all_image_names.extend(names)
        all_labels.extend(labels.cpu().numpy())

    # Concatenate activations from all batches
    for name in layer_names:
        if acts_dict[name]: # Check if list is not empty
             try:
                 acts_dict[name] = np.concatenate(acts_dict[name], axis=0)
             except ValueError as e:
                 print(f"Error concatenating features for layer {name}: {e}")
                 # Decide how to handle: maybe return empty or raise error
                 # For now, set to empty array
                 acts_dict[name] = np.array([])
        else:
            acts_dict[name] = np.array([]) # Ensure it's an empty array if no batches were processed
            
    print() # Newline after progress indicator
    return acts_dict, all_image_names, all_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['tiny-imagenet', 'imagenet'],
                        help='Dataset to extract features from')
    # Add argument to specify pretrained dataset source
    parser.add_argument('--pretrained_source', type=str, default='none', choices=['none', 'imagenet1k'],
                        help='Specify weights source: none (random) or imagenet1k (pretrained)')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Use the argument for pretrained dataset
    pretrained_dataset = args.pretrained_source
    print(f"Using pretrained_dataset source: {pretrained_dataset}")
    model, layers = get_model_and_layers(args.dataset, pretrained_dataset)
    
    # Configure feature extractor with OmegaConf
    cfg = OmegaConf.create({"return_nodes": layers})
    feature_extractor = configure_feature_extractor(cfg, model)
    feature_extractor.to(device).eval()

    data_cfg = {
        "dataset": args.dataset,
        "batchsize": 512, 
        "num_workers": 8,
        "data_augment": False
    }
    # Pass shuffle=False to get the single 'all' loader
    print("Requesting 'all' split by setting shuffle=False")
    datasets, loaders = get_obj_cls_loader(data_cfg, shuffle=False)

    # Expect a single loader named 'all'
    if 'all' not in loaders:
        raise RuntimeError("Expected a single dataloader named 'all' when shuffle=False, but got: {}\n".format(list(loaders.keys())))
        
    loader = loaders['all']
    print(f"Processing 'all' split with {len(loader.dataset)} images...")
    
    # Extract features directly from the 'all' loader
    combined_features, combined_names, combined_labels = extract_features(feature_extractor, loader, device, layers)

    # Check if features were actually extracted
    if not combined_names:
         print("Warning: No image names were collected during feature extraction. Output file will be empty or incomplete.")
         # Depending on desired behavior, maybe exit or raise error

    # Print shapes after extraction
    for layer in combined_features:
        print(f"{layer}: {combined_features[layer].shape}")

    output_dir = os.path.join("datasets", "obj_cls", args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    # Output filename now depends on the pretrained_source argument
    output_path = os.path.join(output_dir, f"features_pretrained_{pretrained_dataset}.npz")
    
    # Save combined features
    save_dict = {
        'image_names': combined_names,
        'labels': combined_labels
    }
    save_dict.update(combined_features)
    
    # Use savez_compressed for potentially large files
    print(f"Saving features for {len(combined_names)} images to {output_path}")
    np.savez_compressed(output_path, **save_dict)
    print("Features saved successfully.")

if __name__ == '__main__':
    main()