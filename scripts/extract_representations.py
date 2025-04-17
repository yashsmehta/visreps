import os
import argparse
import torch
import numpy as np
from omegaconf import OmegaConf
from dotenv import load_dotenv # Import load_dotenv
from visreps.models import standard_cnn
from visreps.models import utils as model_utils # Import model_utils
from visreps.models.utils import configure_feature_extractor
from visreps.dataloaders.obj_cls import get_obj_cls_loader
import json # Import json for loading config
import re # Import re for epoch extraction

# Load environment variables from .env file
load_dotenv()

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
        print(f"Processing batch {batch_idx+1}/{total_batches}", end='\\r')
        images = images.to(device)

        # Get image names from dataset - Adjusted for potentially empty batches or lists
        start = batch_idx * loader.batch_size
        end = start + images.size(0)
        # Check if dataset.samples exists and is indexable
        if hasattr(loader.dataset, 'samples') and loader.dataset.samples and len(loader.dataset.samples) > start:
            # Ensure end index doesn't exceed length
            current_end = min(end, len(loader.dataset.samples))
            # Assuming samples format is (filepath, label) or (filepath, label, metadata...)
            # We need the filepath, which is usually the first element
            try:
                names = [os.path.basename(loader.dataset.samples[i][0]) for i in range(start, current_end)]
            except (TypeError, IndexError):
                 print(f"Warning: Could not retrieve image names for batch {batch_idx+1} from dataset.samples. Check structure.")
                 names = [f"image_{i}" for i in range(start, end)] # Placeholder names
        else:
            # Fallback or alternative way to get names if samples structure is different or unavailable
            # This part might need adjustment based on the exact dataset structure
            # If image paths aren't stored in `samples`, we might need another way.
            # For now, setting to empty if samples[i][0] isn't the path.
            print(f"Warning: Could not retrieve image names for batch {batch_idx+1}. Check dataset structure or availability of samples.")
            names = [f"image_{i}" for i in range(start, end)] # Placeholder names


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
        # Extend with the label tensor itself, convert later
        all_labels.append(labels.cpu()) # Store tensors

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

    # Concatenate labels into a single numpy array
    if all_labels: # Ensure list is not empty
        try:
            # Convert tensors to numpy arrays *before* concatenating
            final_labels = np.concatenate([lbl.numpy().reshape(-1) for lbl in all_labels], axis=0)
        except ValueError as e:
            print(f"Warning: Could not concatenate labels: {e}. Saving as list.")
            # Convert to list of numpy arrays as fallback if concat fails
            final_labels = [lbl.numpy() for lbl in all_labels]
    else:
        final_labels = np.array([]) # Empty array if no labels

    return acts_dict, all_image_names, final_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['tiny-imagenet', 'imagenet'],
                        help='Dataset to extract features from')
    parser.add_argument('--load_from', type=str, required=True, choices=['standard', 'checkpoint'],
                        help="Load model from 'standard' (pretrained/random) or 'checkpoint'.")
    # Arguments for standard model loading
    parser.add_argument('--pretrained_dataset', type=str, default='imagenet1k', choices=['imagenet1k', 'none'],
                        help="Specify weights source for standard AlexNet: 'imagenet1k' (pretrained) or 'none' (random). Used only if --load_from standard.")
    # Arguments for checkpoint loading
    parser.add_argument('--exp_name', type=str, help="Experiment name for checkpoint loading. Required if --load_from checkpoint.")
    parser.add_argument('--cfg_id', type=str, help="Config ID for checkpoint loading. Required if --load_from checkpoint.")
    parser.add_argument('--checkpoint_model', type=str, default='checkpoint_epoch_0.pth',
                        help="Checkpoint filename (e.g., 'checkpoint_epoch_10.pth', 'best_model.pth'). Required if --load_from checkpoint.")
    # Argument for specifying layers
    parser.add_argument('--layers', type=str, required=True,
                        help='Comma-separated list of layer names to extract features from (e.g., "conv1,fc2").')

    args = parser.parse_args()
    cfg_cli = OmegaConf.create(vars(args)) # Create OmegaConf from CLI args

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layers = [layer.strip() for layer in args.layers.split(',')] # Parse layers string

    output_dir = None # Initialize output_dir
    output_filename = None # Initialize output_filename
    model = None # Initialize model
    cfg = None # Initialize cfg

    if args.load_from == 'standard':
        print(f"Loading standard AlexNet model (pretrained: {args.pretrained_dataset})...")
        num_classes = 200 if args.dataset == "tiny-imagenet" else 1000
        # Create a config for standard model loading and feature extraction
        cfg = OmegaConf.create({
            "model_class": "standard_cnn",
            "model_name": "AlexNet",
            "pretrained_dataset": args.pretrained_dataset,
            "return_nodes": layers,
            "load_model_from": "standard", # Explicitly set for clarity
            "pca_labels_folder": "pca_labels" # Add default pca_labels_folder
        })
        # Load the base model first
        model = model_utils.load_model(cfg, device, num_classes=num_classes)
        
        # Define output path for standard loading
        output_dir = os.path.join("datasets", "obj_cls", args.dataset)
        output_filename = f"features_standard_pretrained_{args.pretrained_dataset}.npz"

    elif args.load_from == 'checkpoint':
        if not args.exp_name or not args.cfg_id:
            raise ValueError("--exp_name and --cfg_id are required when --load_from is 'checkpoint'.")
        print(f"Loading model from checkpoint: exp_name={args.exp_name}, cfg_id={args.cfg_id}, checkpoint={args.checkpoint_model}...")

        # Load the training config first to get model details
        config_path = f"model_checkpoints/{args.exp_name}/{args.cfg_id}/config.json"
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, 'r') as f:
            train_cfg_dict = json.load(f)
        cfg_train = OmegaConf.create(train_cfg_dict)

        # Create the config structure expected by load_model when loading from checkpoint
        cfg = OmegaConf.create({
            "load_model_from": "checkpoint",
            "exp_name": args.exp_name,
            "cfg_id": args.cfg_id,
            "checkpoint_model": args.checkpoint_model,
            "return_nodes": layers # Add return_nodes from CLI
        })

        # Merge the essential parts from the loaded training config if they exist
        # This might include model_class, model_name, etc., if needed by subsequent steps
        # or if load_model needs them even when loading from checkpoint (check its implementation)
        # For now, let's assume load_model uses the checkpoint's config primarily
        # If needed, we can merge more selectively: OmegaConf.merge(cfg, cfg_train)

        # Load the model using the checkpoint config
        # load_model handles merging the config from the checkpoint internally
        model = model_utils.load_model(cfg, device)
        
        # Define output path for checkpoint loading
        checkpoint_base_dir = f"model_checkpoints/{args.exp_name}/{args.cfg_id}"
        output_dir = os.path.join(checkpoint_base_dir, "model_representations")
        
        # Extract epoch number from checkpoint filename
        epoch_match = re.search(r'epoch_(\d+)', args.checkpoint_model)
        if epoch_match:
            epoch_number = epoch_match.group(1)
            output_filename = f"model_reps_epoch_{epoch_number}.npz"
        else:
            # Try extracting from 'best_model.pth' or similar generic names if needed
            # For now, raise error if specific epoch format isn't found
            # Consider adding a check for 'best_model.pth' and naming accordingly if needed
            raise ValueError(f"Could not extract epoch number from checkpoint filename: {args.checkpoint_model}. Expected format like 'epoch_<number>.pth'.")


    else:
        raise ValueError(f"Invalid --load_from value: {args.load_from}")

    # Ensure model and cfg are loaded
    if model is None or cfg is None:
         raise RuntimeError("Model or Config was not loaded correctly.")

    # Configure feature extractor using the loaded model and config
    feature_extractor = model_utils.configure_feature_extractor(cfg, model)
    feature_extractor.to(device).eval() # Ensure model is in eval mode and on correct device

    data_cfg = {
        "dataset": args.dataset,
        "batchsize": 512,
        "num_workers": 8,
        "data_augment": False,
        "pca_labels_folder": "pca_labels" # Add pca_labels_folder here
    }
    # Pass shuffle=False to get the single 'all' loader
    print("Requesting 'all' split by setting shuffle=False")
    datasets, loaders = get_obj_cls_loader(data_cfg, shuffle=False)

    # Expect a single loader named 'all'
    if 'all' not in loaders:
        raise RuntimeError("Expected a single dataloader named 'all' when shuffle=False, but got: {}\\n".format(list(loaders.keys())))

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
    # Print label shape too
    if isinstance(combined_labels, np.ndarray):
        print(f"labels: {combined_labels.shape}")
    else:
        print(f"labels: list of length {len(combined_labels)}")


    # Create output directory and define full path
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)


    # Save combined features
    save_dict = {
        'image_names': combined_names,
        'labels': combined_labels # Already concatenated numpy array
    }
    save_dict.update(combined_features)

    # Use savez_compressed for potentially large files
    print(f"Saving features for {len(combined_names)} images to {output_path}")
    np.savez_compressed(output_path, **save_dict)
    print("Features saved successfully.")

if __name__ == '__main__':
    main()