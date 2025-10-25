import timm
import torch
import numpy as np
import os
from tqdm import tqdm
from visreps.dataloaders.obj_cls import get_obj_cls_loader

DATASET = "imagenet"

model = timm.create_model('vit_large_patch16_dinov3', pretrained=True, num_classes=0)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# x = torch.randn(1, 3, 224, 224)

data_cfg = {
    "dataset": DATASET,
    "batchsize": 256,
    "num_workers": 16,
    "data_augment": False,
    "pca_labels_folder": "N/A"
}

datasets, loaders = get_obj_cls_loader(data_cfg, shuffle=False)
loader = loaders['all']
dataset = loader.dataset

features_list = []
image_names_list = []
data_config = timm.data.resolve_model_data_config(model)

# Create the transformation function using the config
preprocess = timm.data.create_transform(**data_config, is_training=False)
dataset.transform = preprocess

with torch.no_grad():
    for batch_idx, (images, _) in enumerate(tqdm(loader, desc="Extracting DINOv3 features", unit="batch")):
        images = images.to(device)
        features = model.forward_features(images)[:,0,:] # use the first token (CLS token)

        # Get image names from dataset samples
        batch_image_names = []
        batch_size = images.shape[0]
        start_idx = batch_idx * loader.batch_size
        for i in range(batch_size):
            idx = start_idx + i
            if idx < len(dataset.samples):
                img_name = dataset.samples[idx][2]
                batch_image_names.append(img_name)

        image_names_list.extend(batch_image_names)
        features_list.append(features.cpu())

# Concatenate all features
all_features = torch.cat(features_list, dim=0).numpy()
print(f"Extracted features shape: {all_features.shape}")
print(f"Collected {len(image_names_list)} image names")

# Ensure we collected image names for all samples
if len(image_names_list) != len(all_features):
    raise ValueError(f"Image names mismatch: collected {len(image_names_list)} names but extracted {len(all_features)} features")

# Save results
output_dir = os.path.join("datasets", "obj_cls", DATASET)
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "features_dino.npz")
np.savez_compressed(output_path, dino_features=all_features, image_names=image_names_list)
print(f"Saved features and image names to {output_path}")