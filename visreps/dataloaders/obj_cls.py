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
    "mnist": [0.1307],
    "cifar10": [0.4914, 0.4822, 0.4465],
}
DS_STD = {
    "tiny-imagenet": [0.272, 0.265, 0.274],
    "imgnet": [0.229, 0.224, 0.225],
    "mnist": [0.3081],
    "cifar10": [0.2023, 0.1994, 0.2010],
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

def get_mnist_transform(data_augment=False):
    """Return a transform for MNIST dataset."""
    transforms_list = []
    if data_augment:
        transforms_list.extend([
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        ])
    transforms_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(DS_MEAN["mnist"], DS_STD["mnist"]),
    ])
    return transforms.Compose(transforms_list)

def get_cifar10_transform(data_augment=False):
    """Return a transform for CIFAR-10 dataset."""
    transforms_list = []
    if data_augment:
        transforms_list.extend([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    transforms_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(DS_MEAN["cifar10"], DS_STD["cifar10"]),
    ])
    return transforms.Compose(transforms_list)

def prepare_mnist_data(cfg: Dict) -> Tuple[Dict, Dict[str, DataLoader]]:
    """Prepare MNIST datasets and dataloaders"""
    base_path = cfg.get("dataset_path") or os.path.join("datasets", "obj_cls", "mnist")
    splits = ["train", "test"]
    
    datasets_dict = {}
    loaders_dict = {}
    
    for split in splits:
        transform = get_mnist_transform(data_augment=(split == "train" and cfg.get("data_augment", True)))
        dataset = datasets.MNIST(
            base_path,
            train=(split == "train"),
            transform=transform,
            download=True
        )
        datasets_dict[split] = dataset
        
        loader = DataLoader(
            dataset,
            batch_size=cfg.get("batchsize", 32),
            shuffle=(split == "train"),
            num_workers=cfg.get("num_workers", 8),
            pin_memory=True
        )
        loaders_dict[split] = loader
        
        print(f"Created {split} dataset with {len(dataset)} samples")
    
    return datasets_dict, loaders_dict

class CIFAR10PCADataset(Dataset):
    """CIFAR10 dataset with PCA-derived labels"""
    def __init__(self, base_dataset, pca_labels_path):
        self.dataset = base_dataset
        self.pca_labels = []  # Store labels in order they appear in CSV
        
        # Load PCA labels from CSV
        with open(pca_labels_path, 'r') as f:
            # Skip header
            header = f.readline()
            # Read label mappings
            for line in f:
                img_name, label = line.strip().split(',')
                self.pca_labels.append(int(label))
                
        if len(self.pca_labels) != len(base_dataset):
            print(f"Warning: Number of PCA labels ({len(self.pca_labels)}) does not match dataset size ({len(base_dataset)})")
                
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, _ = self.dataset[idx]  # Ignore original label
        # Use modulo to handle cases where dataset size doesn't match labels
        pca_label = self.pca_labels[idx % len(self.pca_labels)]
        return img, pca_label

def prepare_cifar10_data(cfg: Dict, pca_labels: bool = False) -> Tuple[Dict, Dict[str, DataLoader]]:
    """Prepare CIFAR-10 datasets and dataloaders"""
    base_path = cfg.get("dataset_path") or os.path.join("datasets", "obj_cls", "cifar10")
    splits = ["train", "test"]
    
    datasets_dict = {}
    loaders_dict = {}
    
    for split in splits:
        transform = get_cifar10_transform(data_augment=(split == "train" and cfg.get("data_augment", True)))
        base_dataset = datasets.CIFAR10(
            base_path,
            train=(split == "train"),
            transform=transform,
            download=True
        )
        
        if pca_labels:  # Apply PCA labels to both train and test
            pca_labels_path = os.path.join(base_path, "pca_labels", "n_bits_3.csv")
            if not os.path.exists(pca_labels_path):
                print(f"Warning: PCA labels file not found at {pca_labels_path}, using original labels")
                dataset = base_dataset
            else:
                dataset = CIFAR10PCADataset(base_dataset, pca_labels_path)
                print(f"Using PCA labels for {split} set")
        else:
            dataset = base_dataset
            
        datasets_dict[split] = dataset
        
        loader = DataLoader(
            dataset,
            batch_size=cfg.get("batchsize", 32),
            shuffle=(split == "train"),
            num_workers=cfg.get("num_workers", 8),
            pin_memory=True
        )
        loaders_dict[split] = loader
        
        print(f"Created {split} dataset with {len(dataset)} samples")
    
    return datasets_dict, loaders_dict

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

def get_obj_cls_loader(cfg, pca_labels=True):
    """
    Prepare object classification datasets and loaders based on config.
    Currently supports 'tiny-imagenet', 'imagenet', 'mnist', and 'cifar10'.
    
    Args:
        cfg: Configuration dictionary
        pca_labels: Whether to use PCA-derived labels (currently only supported for CIFAR-10)
        
    Returns:
        Tuple of (datasets_dict, loaders_dict) where each dict maps split names to 
        corresponding Dataset and DataLoader objects
    """
    dataset = cfg.get("dataset", "tiny-imagenet")
    if dataset == "tiny-imagenet":
        return prepare_tinyimgnet_data(cfg)
    elif dataset == "imagenet":
        return prepare_imgnet_data(cfg)
    elif dataset == "mnist":
        return prepare_mnist_data(cfg)
    elif dataset == "cifar10":
        return prepare_cifar10_data(cfg, pca_labels)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")