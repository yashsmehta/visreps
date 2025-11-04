import warnings

import os
import numpy as np
import torch
from tqdm import tqdm
from visreps.dataloaders.obj_cls import get_obj_cls_loader
warnings.simplefilter("ignore")
from dreamsim import dreamsim

device = "cuda"

model, preprocess = dreamsim(pretrained=True, cache_dir="/home/ymehta3/.cache/dreamsim", device=device)

data_cfg = {
    "dataset": "imagenet",
    "batchsize": 256,
    "num_workers": 16,
    "data_augment": False,
}

datasets, loaders = get_obj_cls_loader(data_cfg, shuffle=False, preprocess=False, train_test_split=False)
loader = loaders['all']
dataset = loader.dataset

image_names_list = [sample[2] for sample in dataset.samples]
dataset.transform = preprocess

features_list = []

with torch.no_grad():
    for images, _ in tqdm(loader, desc="Extracting DreamSim features", unit="batch"):
        images = images.to(device)
        features = model.embed(images)
        features_list.append(features.cpu())

# Concatenate all features
all_features = torch.cat(features_list, dim=0).numpy()
print(f"Extracted features shape: {all_features.shape}")
print(f"Collected {len(image_names_list)} image names")

# Ensure we collected image names for all samples
if len(image_names_list) != len(all_features):
    raise ValueError(f"Image names mismatch: collected {len(image_names_list)} names but extracted {len(all_features)} features")

# Save results
output_dir = os.path.join("datasets", "obj_cls", "imagenet")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "features_dreamsim.npz")
np.savez_compressed(output_path, dreamsim_features=all_features, image_names=image_names_list)
print(f"Saved features and image names to {output_path}")