import os
import argparse
import torch
import numpy as np
from omegaconf import OmegaConf
from dotenv import load_dotenv
from visreps.models import standard_cnn
from visreps.models import utils as model_utils
from visreps.models.utils import configure_feature_extractor
from visreps.dataloaders.obj_cls import get_obj_cls_loader
import re
from tqdm import tqdm

# Load environment variables from .env file
load_dotenv()

def _apply_srp(tensor: torch.Tensor, input_dim: int, srp_dim: int, srp_density: float, device: torch.device, srp_matrices: dict, pbar: tqdm, target_dtype_torch: torch.dtype):
    """Generates (if needed) and applies Sparse Random Projection using PyTorch sparse tensors."""
    if input_dim not in srp_matrices:
        pbar.write(f"  Generating PyTorch sparse SRP matrix (float32, +/-1 values) for Input Dim: {input_dim} -> Target Dim: {srp_dim} on {device}")
        num_elements = int(srp_density * input_dim * srp_dim)
        if num_elements == 0:
            pbar.write(f"[Warning] Calculated zero elements for SRP matrix (dim {input_dim}x{srp_dim}, density {srp_density}). Creating empty sparse matrix.")
            srp_matrix = torch.sparse_coo_tensor(size=(input_dim, srp_dim), dtype=target_dtype_torch, device=device)
        else:
            indices_row = torch.randint(0, input_dim, (num_elements,), device=device)
            indices_col = torch.randint(0, srp_dim, (num_elements,), device=device)
            indices = torch.stack([indices_row, indices_col], dim=0)
            # Use +/- 1 values for SRP matrix entries
            values = (torch.randint(0, 2, (num_elements,), dtype=target_dtype_torch, device=device) * 2 - 1)
            # Optional: Scale values, e.g., by 1/sqrt(num_elements) or 1/sqrt(srp_dim) ?
            # values = values / torch.sqrt(torch.tensor(num_elements, dtype=target_dtype_torch, device=device))

            unique_indices, unique_inverse = torch.unique(indices, dim=1, return_inverse=True)
            # Sum values for duplicate indices (standard practice)
            unique_values = torch.zeros(unique_indices.shape[1], dtype=target_dtype_torch, device=device).scatter_add_(0, unique_inverse, values)
            srp_matrix = torch.sparse_coo_tensor(unique_indices, unique_values, size=(input_dim, srp_dim), dtype=target_dtype_torch)
        srp_matrices[input_dim] = srp_matrix

    R = srp_matrices[input_dim]
    try:
        if tensor.dtype != target_dtype_torch:
            tensor = tensor.to(target_dtype_torch)
        # Sparse mm: R^T @ dense^T -> result^T. So, (D_out x D_in) @ (D_in x B) -> (D_out x B)
        projected_tensor = torch.sparse.mm(R.t(), tensor.t()).t()
        return projected_tensor
    except Exception as e:
        pbar.write(f"[Error] PyTorch sparse MM failed (Input Dim: {input_dim}): {e}")
        # Return None or raise exception to signal failure
        return None # Caller needs to handle None

def extract_features(model, loader, device, layer_names, spatial_pooling, sparse_random_projection):
    """Extracts activations, optionally applying pooling and SRP (using PyTorch sparse tensors).
    Buffers features as CPU tensors and concatenates at the end.
    Returns a dictionary of numpy arrays.
    """
    # Store lists of CPU tensors during processing
    acts_dict_tensor_lists = {name: [] for name in layer_names}
    srp_matrices = {}
    srp_dim = 2048
    srp_density = 0.1
    target_dtype_torch = torch.float32
    target_dtype_np = np.float32

    if len(loader.dataset) == 0:
        print("\n[Warning] Dataset is empty, skipping feature extraction.")
        return {name: np.array([], dtype=target_dtype_np) for name in layer_names}

    total_batches = len(loader)
    if total_batches == 0:
        print("\n[Warning] DataLoader has 0 batches, skipping feature extraction.")
        return {name: np.array([], dtype=target_dtype_np) for name in layer_names}

    pbar = tqdm(total=total_batches, desc="Processing Batches", unit="batch")

    for batch_idx, batch_data in enumerate(loader):
        if isinstance(batch_data, (list, tuple)):
            images = batch_data[0]
        else:
            images = batch_data

        images = images.to(device, dtype=target_dtype_torch)

        with torch.no_grad():
            features = model(images)

        for name in layer_names:
            if name in features:
                feature_map = features[name]
                if feature_map.dtype != target_dtype_torch:
                     pbar.write(f"[Warning] Layer '{name}' feature map has dtype {feature_map.dtype}, casting to {target_dtype_torch}.")
                     feature_map = feature_map.to(target_dtype_torch)

                processed_features_tensor = None

                if name.startswith('conv') and feature_map.ndim == 4:
                    if spatial_pooling:
                        processed_features_tensor = torch.mean(feature_map, dim=(2, 3))
                    else:
                        processed_features_tensor = torch.flatten(feature_map, start_dim=1)
                else:
                    processed_features_tensor = feature_map

                final_tensor_for_batch = None
                if sparse_random_projection:
                    input_dim = processed_features_tensor.shape[1]
                    if input_dim == 0:
                        pbar.write(f"[Warning] Layer '{name}' has 0 features in batch {batch_idx+1}, cannot apply SRP.")
                        # Create tensor of zeros on CPU for this batch
                        final_tensor_for_batch = torch.zeros((processed_features_tensor.shape[0], srp_dim), dtype=target_dtype_torch, device='cpu')
                    else:
                        projected_tensor = _apply_srp(processed_features_tensor, input_dim, srp_dim, srp_density, device, srp_matrices, pbar, target_dtype_torch)
                        if projected_tensor is not None:
                            # Move result to CPU for buffering
                            final_tensor_for_batch = projected_tensor.cpu()
                        else:
                            # Handle SRP failure - create NaNs on CPU
                            pbar.write(f"[Warning] Filling NaNs for layer '{name}' batch {batch_idx+1} due to SRP error.")
                            final_tensor_for_batch = torch.full((processed_features_tensor.shape[0], srp_dim), float('nan'), dtype=target_dtype_torch, device='cpu')
                else:
                    # Move to CPU for buffering if SRP not applied
                    final_tensor_for_batch = processed_features_tensor.cpu()

                # Append the CPU tensor to the list for this layer
                if final_tensor_for_batch is not None:
                     acts_dict_tensor_lists[name].append(final_tensor_for_batch)

            else:
                 pbar.write(f"[Warning] Layer '{name}' not found in model output for batch {batch_idx+1}")

        pbar.update(1)
        pbar.set_postfix({"Pooling": spatial_pooling, "SRP": sparse_random_projection})

    pbar.close()

    # Concatenate tensors and convert to numpy after the loop
    final_acts_dict = {}
    print("\nConcatenating extracted features...")
    for name, tensor_list in tqdm(acts_dict_tensor_lists.items(), desc="Concatenating Layers"):
        if tensor_list:
            try:
                # Concatenate tensors on CPU
                concatenated_tensor = torch.cat(tensor_list, dim=0)
                # Convert final tensor to numpy
                final_acts_dict[name] = concatenated_tensor.numpy().astype(target_dtype_np)
                # Check for NaNs introduced by SRP errors
                if np.isnan(final_acts_dict[name]).any():
                     print(f"  [Warning] NaNs detected in final features for layer '{name}'.")
            except Exception as e:
                print(f"\n[Error] Concatenating features for layer '{name}': {e}")
                shapes = [t.shape for t in tensor_list]
                print(f"  Layer '{name}' tensor shapes in list: {shapes}")
                final_acts_dict[name] = np.array([], dtype=target_dtype_np)
        else:
            final_acts_dict[name] = np.array([], dtype=target_dtype_np)

    return final_acts_dict

def main():
    parser = argparse.ArgumentParser(description="Extract features from a trained model.")
    # --- Argument Parsing (Keep as is) ---
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
    parser.add_argument('--spatial_pooling', action='store_true', default=False,
                        help="Enable spatial pooling (mean) for convolutional layers. Default is to flatten.")
    parser.add_argument('--no_sparse_random_projection', action='store_false', dest='sparse_random_projection',
                        help="Disable sparse random projection. Default is to project.")
    parser.set_defaults(sparse_random_projection=True)

    args = parser.parse_args()

    print("\n--- Feature Extraction Configuration ---")
    print(f"Dataset: {args.dataset}")
    print(f"Load Model From: {args.load_from}")
    if args.load_from == 'standard':
        print(f"Standard Model Source: {args.pretrained_dataset}")
    elif args.load_from == 'checkpoint':
        print(f"Experiment Name: {args.exp_name}")
        print(f"Config ID: {args.cfg_id}")
        print(f"Checkpoint File: {args.checkpoint_model}")
    print(f"Spatial Pooling (Conv): {args.spatial_pooling}")
    print(f"Sparse Random Projection: {args.sparse_random_projection}")

    all_layers_to_extract = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc1', 'fc2']
    print(f"Target Layers: {all_layers_to_extract}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Setup Dataloader ---
    print("\n--- Preparing Dataset --- ")
    data_cfg = {
        "dataset": args.dataset,
        "batchsize": 512,
        "num_workers": 8,
        "data_augment": False,
        "pca_labels_folder": "N/A"
    }
    # Pass shuffle=True if order doesn't matter, False if it does (though names/labels aren't collected)
    # Sticking with False for consistency unless there's a reason to change.
    datasets, loaders = get_obj_cls_loader(data_cfg, shuffle=False)
    if 'all' not in loaders:
        raise RuntimeError(f"[Error] Expected a single dataloader named 'all', but got: {list(loaders.keys())}")
    loader = loaders['all']
    print(f"Processing 'all' split with {len(loader.dataset)} images using {len(loader)} batches...")

    # --- Configure Model ---
    print("\n--- Configuring Model --- ")
    model = None
    cfg = None
    feature_extractor = None
    output_filename = None
    base_output_dir = None

    if args.dataset == "tiny-imagenet": num_classes = 200
    elif args.dataset in ["imagenet", "imagenet-mini-50"]: num_classes = 1000
    else: raise ValueError(f"[Error] Unsupported dataset for class number determination: {args.dataset}")

    if args.load_from == 'standard':
        print(f"Loading standard AlexNet model (source: {args.pretrained_dataset})...")
        cfg = OmegaConf.create({
            "model_class": "standard_cnn", "model_name": "AlexNet",
            "pretrained_dataset": args.pretrained_dataset,
            "return_nodes": all_layers_to_extract,
            "load_model_from": args.load_from,
            "pca_labels_folder": "N/A"
        })
        model = model_utils.load_model(cfg, device, num_classes=num_classes)
        base_output_dir = os.path.join("datasets", "obj_cls", args.dataset)
        output_filename = f"features_alexnet_pretrained_{args.pretrained_dataset}.npz"
        print("Standard model loaded.")

    elif args.load_from == 'checkpoint':
        if not args.exp_name or args.cfg_id is None:
            raise ValueError("[Error] --exp_name and --cfg_id are required when load_from is 'checkpoint'.")
        cfg_dir_name = f"cfg{args.cfg_id}"
        checkpoint_path = f"model_checkpoints/{args.exp_name}/{cfg_dir_name}/{args.checkpoint_model}"
        config_path = f"model_checkpoints/{args.exp_name}/{cfg_dir_name}/config.json"
        print(f"Loading model from checkpoint: {checkpoint_path}...")
        if not os.path.exists(config_path): raise FileNotFoundError(f"[Error] Config file not found: {config_path}")
        if not os.path.exists(checkpoint_path): raise FileNotFoundError(f"[Error] Checkpoint file not found: {checkpoint_path}")

        cfg = OmegaConf.create({
            "load_model_from": args.load_from, "exp_name": args.exp_name, "cfg_id": args.cfg_id,
            "checkpoint_model": args.checkpoint_model,
            "return_nodes": all_layers_to_extract
        })
        model = model_utils.load_model(cfg, device)
        base_output_dir = os.path.join(f"model_checkpoints/{args.exp_name}/{cfg_dir_name}", "model_representations")
        # Use a more flexible regex: ignore case, allow underscore or hyphen
        epoch_match = re.search(r'epoch[_-](\d+)', args.checkpoint_model, re.IGNORECASE)
        if epoch_match:
            epoch_number = epoch_match.group(1)
            output_filename = f"model_reps_epoch_{epoch_number}.npz"
        else:
            checkpoint_name_no_ext = os.path.splitext(args.checkpoint_model)[0]
            output_filename = f"model_reps_{checkpoint_name_no_ext}.npz"
            print(f"[Warning] Could not extract epoch number from {args.checkpoint_model}. Using filename: {output_filename}")
        print("Checkpoint model loaded.")

    else:
        raise ValueError(f"[Error] Invalid load_from value: {args.load_from}")

    if model is None: raise RuntimeError("[Error] Model loading failed.")
    if output_filename is None or base_output_dir is None:
        raise RuntimeError("[Error] Output path or filename determination failed.")

    print("Configuring feature extractor...")
    feature_extractor = model_utils.configure_feature_extractor(cfg, model)
    if feature_extractor is None: raise RuntimeError("[Error] Feature extractor configuration failed.")
    feature_extractor.to(device).eval()
    print("Feature extractor ready.")

    # --- Extract Features ---
    print("\n--- Extracting Features --- ")
    # Update call to extract_features - it now only returns the dictionary
    all_features_dict = extract_features(
        feature_extractor, loader, device, all_layers_to_extract,
        args.spatial_pooling, args.sparse_random_projection
    )

    # --- Save Results ---
    print("\n--- Saving Results --- ")
    # Check if features exist
    features_exist = any(data.size > 0 for data in all_features_dict.values() if isinstance(data, np.ndarray))

    if not features_exist:
         print("[Warning] No features extracted for any layer. Skipping save.")
    else:
        os.makedirs(base_output_dir, exist_ok=True)
        output_path = os.path.join(base_output_dir, output_filename)
        print(f"Output file: {output_path}")

        # Save dictionary only contains features and metadata now
        save_dict = {**all_features_dict}
        save_dict['spatial_pooling'] = int(args.spatial_pooling)
        save_dict['sparse_random_projection'] = int(args.sparse_random_projection)

        # Print metadata and layer shapes
        print(f"Metadata: Spatial Pooling={save_dict['spatial_pooling']}, SRP={save_dict['sparse_random_projection']}")
        for name, data in all_features_dict.items():
            print(f"  Layer '{name}': shape {data.shape}")

        np.savez_compressed(output_path, **save_dict)
        print("\nFeatures saved successfully.")

    # --- Cleanup ---
    print("\n--- Cleaning Up --- ")
    del feature_extractor, model, cfg, all_features_dict
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("Cleared CUDA cache.")
    print("Feature extraction complete.")

if __name__ == '__main__':
    main()