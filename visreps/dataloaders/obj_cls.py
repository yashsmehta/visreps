import os
import json
from typing import Dict, Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image

DS_MEAN = {
    "tiny-imagenet": [0.480, 0.448, 0.398],
    "imgnet": [0.485, 0.456, 0.406],
}
DS_STD = {
    "tiny-imagenet": [0.272, 0.265, 0.274],
    "imgnet": [0.229, 0.224, 0.225],
}

def get_transform(ds_stats="imgnet", data_augment=False, image_size=224):
    """Return a transform composed of resizing, (optional) augmentation, and normalization."""
    transforms_list = [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),
    ]
    if data_augment:
        transforms_list += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)]
    transforms_list += [
        transforms.ToTensor(),
        transforms.Normalize(DS_MEAN[ds_stats], DS_STD[ds_stats]),
    ]
    return transforms.Compose(transforms_list)

class ImageNetDataset(Dataset):
    """Dataset class for ImageNet with flat folder structure"""
    def __init__(self, base_path: str, split: str = "train", transform=None, train_ratio: float = 0.8):
        self.transform = transform
        
        # Load pre-generated label mappings
        label_path = "datasets/obj_cls/imagenet"
        with open(os.path.join(label_path, "folder_labels.json"), "r") as f:
            self.folder_labels = json.load(f)
        
        # Get all image paths and their labels
        self.samples = []
        skipped_folders = set()
        
        for folder_name in os.listdir(base_path):
            if not folder_name.startswith('n'):  # Skip non-class folders
                continue
                
            folder_path = os.path.join(base_path, folder_name)
            if not os.path.isdir(folder_path):
                continue
            
            # Use folder-based label
            if folder_name not in self.folder_labels:
                skipped_folders.add(folder_name)
                continue
                
            label = int(self.folder_labels[folder_name])  # Convert to int for consistency
            
            # Get all images in this class folder
            for img_name in os.listdir(folder_path):
                if img_name.endswith(('.JPEG', '.jpeg', '.jpg')):
                    img_path = os.path.join(folder_path, img_name)
                    self.samples.append((img_path, label))
        
        if skipped_folders:
            print(f"Warning: Skipped {len(skipped_folders)} folders not found in folder labels:")
            print(f"Example folders: {list(skipped_folders)[:5]}")
        
        print(f"Found {len(self.samples)} valid images across {len(set(x[1] for x in self.samples))} classes")
        
        # Split into train/test
        total_size = len(self.samples)
        train_size = int(total_size * train_ratio)
        indices = torch.randperm(total_size).tolist()
        
        if split == "train":
            self.samples = [self.samples[i] for i in indices[:train_size]]
        else:  # test/val split
            self.samples = [self.samples[i] for i in indices[train_size:]]
            
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class TinyImageNetDataset(Dataset):
    """Dataset class for Tiny ImageNet"""
    def __init__(self, base_path: str, split: str, transform=None):
        self.split_folder = "train" if split == "train" else "val"
        self.dataset = datasets.ImageFolder(
            os.path.join(base_path, self.split_folder), 
            transform=transform
        )
        
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int):
        return self.dataset[idx]

def create_tinyimgnet_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 8,
    shuffle: bool = True
) -> DataLoader:
    """Create DataLoader for Tiny ImageNet dataset"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=2,
        pin_memory=True
    )

def create_imgnet_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 8,
    shuffle: bool = True
) -> DataLoader:
    """Create DataLoader for ImageNet dataset"""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=2,
        pin_memory=True
    )

def prepare_tinyimgnet_data(cfg: Dict) -> Tuple[Dict, Dict[str, DataLoader]]:
    """Prepare Tiny ImageNet datasets and dataloaders"""
    base_path = cfg.get("dataset_path") or os.path.join("datasets", "obj_cls", "tiny-imagenet-200")
    splits = ["train", "test"]
    
    # Create datasets
    datasets_dict = {}
    loaders_dict = {}
    
    for split in splits:
        transform = get_transform(
            ds_stats=cfg.get("ds_stats", "tiny-imagenet"),
            data_augment=(split == "train" and cfg.get("data_augment", True)),
            image_size=224
        )
        
        dataset = TinyImageNetDataset(base_path, split, transform)
        datasets_dict[split] = dataset
        
        loader = create_tinyimgnet_dataloader(
            dataset,
            batch_size=cfg.get("batchsize", 32),
            num_workers=cfg.get("num_workers", 8),
            shuffle=(split == "train")
        )
        loaders_dict[split] = loader
    
    return datasets_dict, loaders_dict

def prepare_imgnet_data(cfg: Dict) -> Tuple[Dict, Dict[str, DataLoader]]:
    """Prepare ImageNet datasets and dataloaders"""
    base_path = cfg.get("dataset_path") or "/data/shared/datasets/imagenet"
    splits = ["train", "test"]
    
    # Create datasets
    datasets_dict = {}
    loaders_dict = {}
    
    for split in splits:
        transform = get_transform(
            ds_stats="imgnet",
            data_augment=(split == "train" and cfg.get("data_augment", True)),
            image_size=224
        )
        
        dataset = ImageNetDataset(
            base_path, 
            split=split, 
            transform=transform,
            train_ratio=cfg.get("train_ratio", 0.8)
        )
        datasets_dict[split] = dataset
        
        loader = create_imgnet_dataloader(
            dataset,
            batch_size=cfg.get("batchsize", 32),
            num_workers=cfg.get("num_workers", 8),
            shuffle=(split == "train")
        )
        loaders_dict[split] = loader
        
        print(f"Created {split} dataset with {len(dataset)} samples")
    
    return datasets_dict, loaders_dict

def get_obj_cls_loader(cfg: Dict) -> Tuple[Dict, Dict[str, DataLoader]]:
    """
    Prepare object classification datasets and loaders based on config.
    Currently supports 'tiny-imagenet' and 'imagenet'.
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        Tuple of (datasets_dict, loaders_dict) where each dict maps split names to 
        corresponding Dataset and DataLoader objects
    """
    dataset = cfg.get("dataset", "tiny-imagenet")
    if dataset == "tiny-imagenet":
        return prepare_tinyimgnet_data(cfg)
    elif dataset == "imagenet":
        return prepare_imgnet_data(cfg)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")