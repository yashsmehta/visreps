#!/usr/bin/env python3
"""
Minimal script to extract CLIP representations from ImageNet datasets.
"""

import os
import argparse
import torch
import numpy as np
import clip
from dotenv import load_dotenv
from tqdm import tqdm

from visreps.dataloaders.obj_cls import get_obj_cls_loader
import visreps.utils as utils

load_dotenv()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Extract CLIP representations from ImageNet datasets")
    parser.add_argument("--dataset", type=str, default="imagenet-mini-50",
                       choices=["imagenet", "imagenet-mini-50"],
                       help="Dataset to extract features from (default: imagenet-mini-50)")
    
    args = parser.parse_args()
    
    # Load CLIP model and preprocessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    print(f"Loaded CLIP model on {device}")
    
    # Setup dataset configuration
    data_cfg = {
        "dataset": args.dataset,
        "batchsize": 512,
        "num_workers": 8,
        "data_augment": False,
        "pca_labels_folder": "N/A"
    }
    
    print(f"Loading dataset: {args.dataset}")
    datasets, loaders = get_obj_cls_loader(data_cfg, shuffle=False)
    if 'all' not in loaders:
        raise RuntimeError(f"Expected a single dataloader named 'all', but got: {list(loaders.keys())}")
    loader = loaders['all']
    dataset = loader.dataset
    
    # Override the dataset's transform with CLIP's preprocessor
    dataset.transform = preprocess
    print(f"Using CLIP preprocessor for feature extraction")
    
    print(f"Loaded {len(dataset)} images from {args.dataset}")
    
    features_list = []
    image_names_list = []
    
    with torch.no_grad():
        for batch_idx, (images, _) in enumerate(tqdm(loader, desc="Extracting CLIP features", unit="batch")):
            # Collect image names for this batch
            batch_size = images.shape[0]
            start_idx = batch_idx * loader.batch_size
            end_idx = min(start_idx + batch_size, len(dataset))
            
            # Get image names from dataset samples
            batch_image_names = []
            for idx in range(start_idx, end_idx):
                if hasattr(dataset, 'samples') and idx < len(dataset.samples):
                    # For ImageNetDataset, samples[idx][2] contains the image name
                    img_name = dataset.samples[idx][2]
                    batch_image_names.append(img_name)
            
            image_names_list.extend(batch_image_names)
            
            images = images.to(device)
            image_features = model.encode_image(images)
            # Normalize features (CLIP standard)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            features_list.append(image_features.cpu())
    
    # Concatenate all features
    all_features = torch.cat(features_list, dim=0).numpy()
    print(f"Extracted features shape: {all_features.shape}")
    print(f"Collected {len(image_names_list)} image names")
    
    # Save results
    output_dir = os.path.join("datasets", "obj_cls", args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "features_clip_vit.npz")
    
    np.savez_compressed(output_path, clip_features=all_features, image_names=image_names_list)
    print(f"Saved features and image names to {output_path}")

if __name__ == '__main__':
    main()
