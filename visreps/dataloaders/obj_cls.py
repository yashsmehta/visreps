import os
from typing import Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets

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

def get_obj_cls_loader(cfg: Dict) -> Tuple[Dict, Dict[str, DataLoader]]:
    """
    Prepare object classification datasets and loaders based on config.
    Currently supports only 'tiny-imagenet'.
    
    Args:
        cfg: Configuration dictionary
        
    Returns:
        Tuple of (datasets_dict, loaders_dict) where each dict maps split names to 
        corresponding Dataset and DataLoader objects
    """
    dataset = cfg.get("dataset", "tiny-imagenet")
    if dataset != "tiny-imagenet":
        raise ValueError(f"Unknown dataset: {dataset}")

    return prepare_tinyimgnet_data(cfg)