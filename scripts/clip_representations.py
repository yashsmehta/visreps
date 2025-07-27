#!/usr/bin/env python3
"""
Minimal script to extract CLIP representations from ImageNet.
"""

import os
import torch
import numpy as np
import clip
from dotenv import load_dotenv

# Import existing functionality
from visreps.dataloaders.obj_cls import ImageNetDataset, create_dataloader
import visreps.utils as utils

# Load environment variables
load_dotenv()

def main():
    # Load CLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    print(f"Loaded CLIP model on {device}")
    
    # Setup ImageNet dataset with CLIP preprocessor
    base_path = utils.get_env_var("IMAGENET_DATA_DIR")
    dataset = ImageNetDataset(base_path, split="all", transform=preprocess)
    loader = create_dataloader(dataset, batch_size=32, num_workers=4, shuffle=False)
    print(f"Loaded {len(dataset)} ImageNet images")
    
    # Extract features
    print("Extracting CLIP features...")
    features_list = []
    
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            image_features = model.encode_image(images)
            # Normalize features (CLIP standard)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            features_list.append(image_features.cpu())
    
    # Concatenate all features
    all_features = torch.cat(features_list, dim=0).numpy()
    print(f"Extracted features shape: {all_features.shape}")
    
    # Save results
    output_dir = os.path.join("datasets", "obj_cls", "imagenet")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "features_clip_vit_b_32.npz")
    
    np.savez_compressed(output_path, clip_features=all_features)
    print(f"Saved features to {output_path}")

if __name__ == '__main__':
    main()
