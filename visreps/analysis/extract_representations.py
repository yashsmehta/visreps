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
    """Extracts activations using the feature extractor for the specified layers."""
    acts_dict = {name: [] for name in layer_names}
    all_image_names, all_labels = [], []
    first_batch = True # Flag to track if we need image names/labels

    if len(loader.dataset) == 0:
        print("Warning: Dataset is empty, skipping feature extraction.")
        for name in layer_names:
            acts_dict[name] = np.array([])
        return acts_dict, [], [], False # Return names_labels_extracted=False

    total_batches = len(loader)
    if total_batches == 0:
        print("Warning: DataLoader has 0 batches, skipping feature extraction.")
        for name in layer_names:
            acts_dict[name] = np.array([])
        return acts_dict, [], [], False # Return names_labels_extracted=False

    for batch_idx, (images, labels) in enumerate(loader):
        print(f"Processing batch {batch_idx+1}/{total_batches}", end='\r')
        images = images.to(device)

        # Only get image names and labels on the first pass through the data
        if first_batch:
            start = batch_idx * loader.batch_size
            end = start + images.size(0)
            if hasattr(loader.dataset, 'samples') and loader.dataset.samples and len(loader.dataset.samples) > start:
                current_end = min(end, len(loader.dataset.samples))
                try:
                    names = [os.path.basename(loader.dataset.samples[i][0]) for i in range(start, current_end)]
                except (TypeError, IndexError):
                     print(f"Warning: Could not retrieve image names for batch {batch_idx+1}. Using placeholders.")
                     names = [f"image_{i}" for i in range(start, end)]
            else:
                print(f"Warning: Could not retrieve image names for batch {batch_idx+1}. Using placeholders.")
                names = [f"image_{i}" for i in range(start, end)]
            all_image_names.extend(names)
            all_labels.append(labels.cpu()) # Store tensors

        with torch.no_grad():
            features = model(images)  # Returns dict of layer activations

        for name in layer_names:
            if name in features:
                # Convert to numpy and handle potential spatial averaging
                feature_map = features[name].cpu().numpy()
                # Check if it's a conv layer (4 dims: B, C, H, W) and average spatial dims
                if name.startswith('conv') and feature_map.ndim == 4:
                    # Average over height (axis 2) and width (axis 3)
                    averaged_features = np.mean(feature_map, axis=(2, 3))
                    acts_dict[name].append(averaged_features)
                else:
                    acts_dict[name].append(feature_map) # Keep as is for non-conv or already averaged features
            else:
                print(f"Warning: Layer {name} not found in model output for batch {batch_idx+1}")

        # After processing the first batch, set the flag to False
        if batch_idx == 0:
            first_batch = False


    # Concatenate activations from all batches
    for name in layer_names:
        if acts_dict[name]:
             try:
                 acts_dict[name] = np.concatenate(acts_dict[name], axis=0)
             except ValueError as e:
                 print(f"Error concatenating features for layer {name}: {e}")
                 acts_dict[name] = np.array([])
        else:
            acts_dict[name] = np.array([])

    # Clear the progress line
    print(" " * 80, end='\r')

    # Concatenate labels into a single numpy array (only if extracted)
    final_labels = np.array([])
    if all_labels:
        try:
            final_labels = np.concatenate([lbl.numpy().reshape(-1) for lbl in all_labels], axis=0)
        except ValueError as e:
            print(f"Warning: Could not concatenate labels: {e}. Labels will be empty.")
            final_labels = np.array([]) # Ensure it's an empty array on failure

    # Return whether names/labels were extracted in this call
    names_labels_extracted = len(all_image_names) > 0 and final_labels.size > 0

    return acts_dict, all_image_names, final_labels, names_labels_extracted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=False, default='imagenet-mini-50',
                        choices=['tiny-imagenet', 'imagenet', 'imagenet-mini-50'],
                        help='Dataset to extract features from (default: imagenet-mini-50)')
    parser.add_argument('--pretrained_dataset', type=str, default='imagenet1k', choices=['imagenet1k', 'none'],
                        help="Specify weights source for standard AlexNet: 'imagenet1k' (pretrained) or 'none' (random). Used only if load_from is standard.")
    parser.add_argument('--exp_name', type=str, default='imagenet_pca', help="Experiment name for checkpoint loading.")
    parser.add_argument('--cfg_id', type=int, default=1, help="Config ID (integer) for checkpoint loading.")
    parser.add_argument('--checkpoint_model', type=str, default='checkpoint_epoch_0.pth',
                        help="Checkpoint filename (e.g., 'checkpoint_epoch_10.pth').")
    parser.add_argument('--load_from', type=str, default='standard', choices=['standard', 'checkpoint'],
                        help="Where to load the model from: 'standard' (pretrained/random) or 'checkpoint'.")

    args = parser.parse_args()

    # --- Hardcoded values ---
    # load_from = 'standard' # Removed hardcoded value
    all_layers_to_extract = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc1', 'fc2']
    # -----------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Setup Dataloader (once) ---
    data_cfg = {
        "dataset": args.dataset,
        "batchsize": 512,
        "num_workers": 8,
        "data_augment": False,
        "pca_labels_folder": "N/A"
    }
    print("Requesting 'all' split by setting shuffle=False")
    datasets, loaders = get_obj_cls_loader(data_cfg, shuffle=False)
    if 'all' not in loaders:
        raise RuntimeError(f"Expected a single dataloader named 'all', but got: {list(loaders.keys())}")
    loader = loaders['all']
    print(f"Processing 'all' split with {len(loader.dataset)} images...")
    # -----------------------------

    # --- Configure Model and Feature Extractor (once for all layers) ---
    print("\n--- Configuring Model --- ")
    model = None
    cfg = None
    feature_extractor = None
    output_filename = None # Define output filename variable
    base_output_dir = None # Define base output directory variable

    if args.load_from == 'standard':
        print(f"Loading standard AlexNet model (pretrained: {args.pretrained_dataset})")
        print(f"Target layers: {all_layers_to_extract}")
        if args.dataset == "tiny-imagenet": num_classes = 200
        elif args.dataset in ["imagenet", "imagenet-mini-50"]: num_classes = 1000
        else: raise ValueError(f"Unsupported dataset: {args.dataset}")

        cfg = OmegaConf.create({
            "model_class": "standard_cnn", "model_name": "AlexNet",
            "pretrained_dataset": args.pretrained_dataset,
            "return_nodes": all_layers_to_extract, # IMPORTANT: All layers now
            "load_model_from": args.load_from, # Use arg value
            "pca_labels_folder": "N/A"
        })
        model = model_utils.load_model(cfg, device, num_classes=num_classes)
        base_output_dir = os.path.join("datasets", "obj_cls", args.dataset)
        # New filename format for standard AlexNet
        output_filename = f"features_alexnet_pretrained_{args.pretrained_dataset}.npz"

    elif args.load_from == 'checkpoint':
        if not args.exp_name or args.cfg_id is None:
            raise ValueError("--exp_name and --cfg_id are required when load_from is 'checkpoint'.")
        cfg_dir_name = f"cfg{args.cfg_id}"
        print(f"Loading model from checkpoint: {args.exp_name}/{cfg_dir_name}/{args.checkpoint_model}")
        print(f"Target layers: {all_layers_to_extract}")
        config_path = f"model_checkpoints/{args.exp_name}/{cfg_dir_name}/config.json"
        if not os.path.exists(config_path): raise FileNotFoundError(config_path)
        with open(config_path, 'r') as f: train_cfg_dict = json.load(f)

        cfg = OmegaConf.create({
            "load_model_from": args.load_from, "exp_name": args.exp_name, "cfg_id": args.cfg_id, # Use arg value
            "checkpoint_model": args.checkpoint_model,
            "return_nodes": all_layers_to_extract # IMPORTANT: All layers now
        })
        model = model_utils.load_model(cfg, device) # Assumes load_model gets num_classes from checkpoint
        checkpoint_base_dir = f"model_checkpoints/{args.exp_name}/{cfg_dir_name}"
        base_output_dir = os.path.join(checkpoint_base_dir, "model_representations")
        epoch_match = re.search(r'epoch_(\d+)', args.checkpoint_model)
        if epoch_match:
            epoch_number = epoch_match.group(1)
            # Keep existing filename format for checkpoints
            output_filename = f"model_reps_epoch_{epoch_number}.npz"
        else:
            raise ValueError(f"Could not extract epoch number from {args.checkpoint_model}")

    else:
        raise ValueError(f"Invalid load_from value: {args.load_from}") # Use arg value

    if model is None: raise RuntimeError("Model was not loaded.")
    if output_filename is None or base_output_dir is None:
        raise RuntimeError("Output path or filename was not determined.")

    feature_extractor = model_utils.configure_feature_extractor(cfg, model)
    feature_extractor.to(device).eval()
    # ----------------------------------------------------------------

    # --- Extract Features for ALL layers at once ---
    print(f"\n--- Extracting Features for Layers: {all_layers_to_extract} ---")
    all_features_dict, extracted_image_names, extracted_labels, names_labels_extracted = extract_features(
        feature_extractor, loader, device, all_layers_to_extract
    )

    # --- Save results for ALL layers into a SINGLE file ---
    if not names_labels_extracted or extracted_image_names is None or extracted_labels is None:
        print("Error: Image names and/or labels were not extracted. Cannot save features.")
    elif not all_features_dict:
        print("Warning: No features extracted for any layer. Skipping save.")
    else:
        os.makedirs(base_output_dir, exist_ok=True)
        output_path = os.path.join(base_output_dir, output_filename)

        # Prepare dict for saving: all layer features + names/labels
        save_dict = {**all_features_dict} # Copy features
        save_dict['image_names'] = extracted_image_names
        save_dict['labels'] = extracted_labels

        print(f"\n--- Saving Results ---")
        print(f"Output file: {output_path}")
        for name, data in all_features_dict.items():
            print(f"  Layer '{name}': shape {data.shape}")
        np.savez_compressed(output_path, **save_dict)
        print("Features saved successfully.")
        # ------------------------------------

    # Clear memory
    del feature_extractor, model, cfg, all_features_dict, extracted_image_names, extracted_labels
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("\n--- Feature Extraction Complete ---")


if __name__ == '__main__':
    main()