import os
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor
import pandas as pd
import re

import visreps.neural_data_processor as neural_data_processor
import visreps.metrics as metrics
from visreps.visreps.dataloaders import get_transform


class StimuliDataset(Dataset):
    def __init__(self, stimuli_dict, transform=None):
        self.transform = transform
        self.keys = sorted(stimuli_dict.keys())
        self.images = [stimuli_dict[k] for k in self.keys]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        key = self.keys[idx]
        img = Image.fromarray(img.astype("uint8"), "RGB")
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, key  # Return key as is


def create_dataloader(stimuli_dict, transform=None, batch_size=32, num_workers=4):
    dataset = StimuliDataset(stimuli_dict, transform)

    def custom_collate_fn(batch):
        images, keys = zip(*batch)
        images = torch.stack(images, 0)
        return images, list(keys)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
    )


def extract_activations(model, dataloader, device):
    model.eval()
    activations_dict = {}
    all_keys = []
    with torch.no_grad():
        for images, keys in dataloader:
            inputs = images.to(device)
            outputs = model(inputs)
            all_keys.extend(keys)  # 'keys' is a list
            for node_name, output in outputs.items():
                output = output.cpu()
                activations_dict.setdefault(node_name, []).append(output)
    for node_name in activations_dict:
        activations_dict[node_name] = torch.cat(activations_dict[node_name], dim=0)
    return activations_dict, all_keys


def eval(cfg):
    """
    Evaluate RSA scores for model checkpoints.
    If cfg.checkpoint_path is a file, evaluates only that checkpoint.
    Otherwise, evaluates all checkpoints in the directory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    neural_data, stimuli = neural_data_processor.load_benchmark_data(cfg)
    transform = get_transform(image_size=64)
    dataloader = create_dataloader(stimuli, transform, batch_size=32, num_workers=4)

    if os.path.isfile(cfg.checkpoint_path):
        checkpoint_files = [cfg.checkpoint_path]
    else:
        checkpoint_dir = cfg.checkpoint_path
        print("Evaluating all checkpoints in:", checkpoint_dir)
        checkpoint_files = sorted(
            glob.glob(os.path.join(checkpoint_dir, "model_epoch_*.pth"))
        )
        if not checkpoint_files:
            raise ValueError(f"No checkpoint files found in {checkpoint_dir}")
        print(f"Found {len(checkpoint_files)} checkpoint files")

    results_list = []
    for checkpoint_file in checkpoint_files:
        print(f"\nEvaluating checkpoint: {checkpoint_file}")

        try:
            model = torch.load(checkpoint_file)
            print("Model loaded directly from checkpoint.")
        except Exception as e:
            print(f"Failed to load model directly from checkpoint: {e}")

        model.to(device)

        return_nodes = list(getattr(cfg, "return_nodes", None))
        if return_nodes:
            if isinstance(return_nodes, list):
                return_nodes = {node: node for node in return_nodes}
            elif isinstance(return_nodes, dict):
                pass  # Use as is
            else:
                raise ValueError("return_nodes should be a list or dict")
            model = create_feature_extractor(model, return_nodes=return_nodes)
        else:
            raise ValueError(
                "No return_nodes specified in the configuration for feature extraction."
            )

        activations_dict, keys = extract_activations(model, dataloader, device)
        print("Activations Dictionary:")
        for node_name, activations in activations_dict.items():
            print(f"{node_name}: {activations.shape}")

        neural_responses = np.array([neural_data[key] for key in keys])
        print("\nNeural Responses:")
        print(f"Shape: {neural_responses.shape}")

        rsa_scores = {}
        for layer, activations in activations_dict.items():
            print(f"Layer: {layer}, Activations Shape: {activations.shape}")
            # Flatten activations if they're not 2D
            if len(activations.shape) > 2:
                activations = activations.view(activations.size(0), -1)
            assert (
                activations.shape[0] == neural_responses.shape[0]
            ), f"Mismatch in number of samples between activations and neural_responses for layer {layer}"
            rsa_scores[layer] = metrics.calculate_rsa_score(
                neural_responses, activations.numpy()
            )

        print(f"RSA Scores: {rsa_scores}")

        # Extract epoch number
        epoch_match = re.search(
            r"model_epoch_(\d+)\.pth", os.path.basename(checkpoint_file)
        )
        epoch = int(epoch_match.group(1)) if epoch_match else None

        for layer, score in rsa_scores.items():
            result = {"epoch": epoch, "layer": layer, "rsa_score": score}
            # Add all cfg attributes to the result
            result.update(vars(cfg))
            results_list.append(result)
        print(f"Epoch {epoch}: RSA Score = {rsa_scores}")

    results_df = pd.DataFrame(results_list)

    # Determine the save directory based on checkpoint_path
    save_dir = (
        os.path.dirname(cfg.checkpoint_path)
        if os.path.isfile(cfg.checkpoint_path)
        else cfg.checkpoint_path
    )

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Update the save path to use the determined directory
    results_path = os.path.join(save_dir, "results.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to: {results_path}")

    return results_df
