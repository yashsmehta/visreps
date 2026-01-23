"""Shared utilities for representation extraction."""

import os
import torch
import numpy as np
from tqdm import tqdm
from dotenv import load_dotenv

from visreps.dataloaders.obj_cls import get_obj_cls_loader

load_dotenv()


def get_loaders(dataset: str, batch_size: int = 256, num_workers: int = 16):
    """Get data loaders for the specified dataset.

    Uses train_test_split=False to load ALL images (not random train/test splits).
    This ensures features are extracted for 100% of the dataset.
    """
    data_cfg = {
        "dataset": dataset,
        "batchsize": batch_size,
        "num_workers": num_workers,
        "data_augment": False,
        "pca_labels_folder": "N/A"
    }
    _, loaders = get_obj_cls_loader(data_cfg, shuffle=False, train_test_split=False)
    return [loaders['all']]


def extract_features(model, loader_list, extract_fn, device, desc="Extracting features"):
    """
    Extract features from a model.

    Args:
        model: The model to extract features from
        loader_list: List of data loaders
        extract_fn: Function that takes (model, images) and returns features
        device: torch device
        desc: Description for progress bar

    Returns:
        features: numpy array of shape (N, D)
        image_names: list of image names
    """
    features_list = []
    image_names_list = []

    with torch.no_grad():
        for loader in loader_list:
            dataset = loader.dataset
            sample_idx = 0
            for images, _ in tqdm(loader, desc=desc, unit="batch"):
                images = images.to(device)
                features = extract_fn(model, images)

                for _ in range(images.shape[0]):
                    if hasattr(dataset, 'samples'):
                        image_names_list.append(dataset.samples[sample_idx][2])
                    sample_idx += 1

                features_list.append(features.cpu())

    all_features = torch.cat(features_list, dim=0).numpy()

    if len(image_names_list) != len(all_features):
        raise ValueError(f"Mismatch: {len(image_names_list)} names vs {len(all_features)} features")

    return all_features, image_names_list


def save_features(features, image_names, dataset: str, model_name: str):
    """Save features to npz file."""
    output_dir = os.path.join("datasets", "obj_cls", dataset)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"features_{model_name}.npz")
    np.savez_compressed(output_path, **{f"{model_name}_features": features, "image_names": image_names})
    print(f"Saved {features.shape} to {output_path}")
