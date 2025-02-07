import os
import json
from typing import Dict, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image
import pandas as pd

# Normalization statistics for each dataset.
DS_MEAN = {
    "tiny-imagenet": [0.480, 0.448, 0.398],
    "imgnet": [0.485, 0.456, 0.406],
    "cifar10": [0.4914, 0.4822, 0.4465],
}
DS_STD = {
    "tiny-imagenet": [0.272, 0.265, 0.274],
    "imgnet": [0.229, 0.224, 0.225],
    "cifar10": [0.2023, 0.1994, 0.2010],
}


def get_transform(ds_stats="imgnet", data_augment=False, image_size=224):
    """
    Create a composed transform with resizing, (optional) augmentation, and normalization.
    
    Args:
        ds_stats: Dataset stats to use for normalization ('imgnet', 'tiny-imagenet', 'cifar10')
        data_augment: Whether to apply data augmentation
        image_size: Target image size. Note: For tiny-imagenet, this is ignored and 64 is used.
    """
    # Handle dataset-specific sizes
    if ds_stats == "tiny-imagenet":
        resize_size = 64
        crop_size = 64
    else:
        resize_size = 256
        crop_size = image_size

    transform_list = [
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(crop_size),
    ]
    if data_augment:
        transform_list += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)]
    transform_list += [
        transforms.ToTensor(),
        transforms.Normalize(DS_MEAN[ds_stats], DS_STD[ds_stats]),
    ]
    return transforms.Compose(transform_list)


# -----------------------------------------------------------------------------
# Generic PCA Dataset wrapper
# -----------------------------------------------------------------------------
class PCADataset(Dataset):
    """
    A dataset wrapper that replaces the original labels with PCA-derived ones.
    The CSV file must have a header with columns:
        - "image": image identifier (filename only)
        - "pca_label": PCA-derived label (integer)
    """
    def __init__(self, base_dataset: Dataset, pca_labels_path: str):
        self.dataset = base_dataset
        self.label_map = self._load_pca_labels(pca_labels_path)
        self._validate_label_coverage()

    def _load_pca_labels(self, pca_labels_path: str) -> dict:
        try:
            df = pd.read_csv(pca_labels_path)
        except Exception as e:
            raise RuntimeError(f"Error reading PCA labels CSV at {pca_labels_path}: {e}")

        required_cols = ["image", "pca_label"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"PCA labels CSV must contain columns: {required_cols}. Missing: {missing_cols}")

        # Ensure PCA labels are integers and non-negative.
        if df["pca_label"].dtype.kind not in "iu":
            raise ValueError("PCA labels must be integers")
        if df["pca_label"].min() < 0:
            raise ValueError("PCA labels must be non-negative")

        label_map = {
            os.path.basename(row["image"]): int(row["pca_label"])
            for _, row in df.iterrows()
        }
        print(f"Loaded {len(label_map)} PCA labels from {pca_labels_path}")
        distribution = pd.Series(list(label_map.values())).value_counts().sort_index().to_dict()
        print(f"Label distribution: {distribution}")
        return label_map

    def _validate_label_coverage(self):
        if hasattr(self.dataset, "samples"):
            dataset_files = {os.path.basename(sample[0]) for sample in self.dataset.samples}
            missing_files = dataset_files - set(self.label_map.keys())
            if missing_files:
                n_missing = len(missing_files)
                examples = list(missing_files)[:3]
                print(f"Warning: {n_missing} files missing PCA labels. Examples: {examples}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]

        if hasattr(self.dataset, "samples"):
            # Use the stored relative path (index 2) as the image identifier.
            img_id = os.path.basename(self.dataset.samples[idx][2])
        else:
            img_id = str(idx)

        label = self.label_map.get(img_id, -1)
        if label == -1:
            raise ValueError(f"No PCA label found for image {img_id}")

        return image, label

# -----------------------------------------------------------------------------
# Dataset classes for ImageNet and Tiny ImageNet
# -----------------------------------------------------------------------------
class ImageNetDataset(Dataset):
    """
    Custom ImageNet loader for a flat folder structure.
    Expects a JSON file mapping folder names to labels (in "datasets/obj_cls/imagenet/folder_labels.json").
    """
    def __init__(self, base_path: str, split: str = "train", transform=None, train_ratio: float = 0.8):
        self.transform = transform
        label_path = os.path.join("datasets", "obj_cls", "imagenet")
        with open(os.path.join(label_path, "folder_labels.json"), "r") as f:
            self.folder_labels = json.load(f)
        self.samples = []
        skipped_folders = set()
        for folder in os.listdir(base_path):
            if not folder.startswith("n"):
                continue
            folder_path = os.path.join(base_path, folder)
            if not os.path.isdir(folder_path):
                continue
            if folder not in self.folder_labels:
                skipped_folders.add(folder)
                continue
            label = int(self.folder_labels[folder])
            for img_name in os.listdir(folder_path):
                if img_name.lower().endswith((".jpeg", ".jpg")):
                    img_path = os.path.join(folder_path, img_name)
                    # Store the folder/image_name as the image identifier
                    img_id = f"{folder}/{img_name}"
                    self.samples.append((img_path, label, img_id))
        if skipped_folders:
            print(f"Warning: Skipped {len(skipped_folders)} folders not found in folder labels. "
                  f"Example: {list(skipped_folders)[:5]}")
        total = len(self.samples)
        train_count = int(total * train_ratio)
        indices = torch.randperm(total).tolist()
        if split == "train":
            self.samples = [self.samples[i] for i in indices[:train_count]]
        else:
            self.samples = [self.samples[i] for i in indices[train_count:]]
        print(f"ImageNet: Loaded {len(self.samples)} samples for split '{split}'.")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        img_path, label, img_id = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


class TinyImageNetDataset(Dataset):
    """
    Tiny ImageNet loader that wraps torchvision's ImageFolder.
    """
    def __init__(self, base_path: str, split: str, transform=None):
        self.split_folder = "train" if split == "train" else "val"
        self.root = os.path.join(base_path, self.split_folder)
        self.dataset = datasets.ImageFolder(self.root, transform=transform)
        self.loader = self.dataset.loader
        self.transform = self.dataset.transform
        self.num_classes = len(self.dataset.classes)  # Get number of classes from dataset
        
        # Debug: Print dataset information
        print(f"\nTinyImageNet {split} Dataset Info:")
        print(f"Root path: {self.root}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Total samples: {len(self.dataset)}")
        
        # Validate class indices
        class_to_idx = self.dataset.class_to_idx
        print(f"Class index range: [{min(class_to_idx.values())}, {max(class_to_idx.values())}]")
        
        # Count samples per class
        class_counts = {}
        for _, label in self.dataset.samples:
            class_counts[label] = class_counts.get(label, 0) + 1
        print(f"Samples per class: min={min(class_counts.values())}, max={max(class_counts.values())}")
        
        # Validate transforms
        if transform is not None:
            print("\nTransform pipeline:")
            for t in transform.transforms:
                print(f"  {t.__class__.__name__}")
        
        # Store original paths for image names
        self.samples = []
        for path, label in self.dataset.samples:
            # Get relative path from dataset root for consistent naming
            rel_path = os.path.relpath(path, self.root)
            self.samples.append((path, label, rel_path))
            
            # Validate label range
            if not (0 <= label < self.num_classes):
                raise ValueError(f"Invalid label {label} for image {path}. Must be in range [0, {self.num_classes})")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        path, label, _ = self.samples[idx]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
            
            # Debug: Validate transformed image (for first few samples)
            if idx < 5:  # Only check first 5 samples to avoid slowdown
                if not isinstance(image, torch.Tensor):
                    raise ValueError(f"Transform did not produce a tensor for image {path}")
                if image.dim() != 3:
                    raise ValueError(f"Expected 3D tensor (C,H,W), got shape {image.shape} for image {path}")
                if image.shape[0] != 3:
                    raise ValueError(f"Expected 3 channels, got {image.shape[0]} for image {path}")
        
        return image, label


# -----------------------------------------------------------------------------
# Helper functions to create/possibly wrap datasets with PCA labels
# -----------------------------------------------------------------------------
def create_collate_fn():
    """Create a simple collate function for (image, label) pairs."""
    def collate_fn(batch):
        images, labels = zip(*batch)
        return torch.stack(images), torch.tensor(labels)
    return collate_fn

def create_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 8,
    shuffle: bool = True,
    collate_fn = None
) -> DataLoader:
    """Create a DataLoader with standard configuration."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=2,
        pin_memory=True,
        collate_fn=collate_fn or create_collate_fn()
    )

def maybe_wrap_with_pca(dataset: Dataset, base_path: str, cfg: Dict, split: str, pca_labels: bool) -> Dataset:
    """
    If pca_labels is True, attempt to wrap the given dataset with PCA labels.
    The CSV file is expected at: <base_path>/pca_labels/n_classes_{pca_n_classes}.csv.
    
    Args:
        dataset: Base dataset to potentially wrap
        base_path: Path to dataset directory
        cfg: Config dict containing:
            - pca_n_classes: Number of PCA-derived classes to use (for file selection)
            - n_classes: Number of output classes for the model
        split: Dataset split ('train' or 'test')
        pca_labels: Whether to use PCA labels
    """
    if pca_labels:
        n_classes = cfg.get("pca_n_classes", 4)  # Use pca_n_classes for file selection
        print(f"Using {n_classes} classes for PCA labels")
        pca_file = f"n_classes_{n_classes}.csv"
        pca_path = os.path.join(base_path, "pca_labels", pca_file)
        if not os.path.exists(pca_path):
            print(f"Warning: PCA labels file not found at {pca_path}. Using original labels.")
            return dataset
        print(f"Using PCA labels for {split} set from {pca_path}")
        return PCADataset(dataset, pca_path)
    return dataset


# -----------------------------------------------------------------------------
# Dataset preparation functions for each supported dataset
# -----------------------------------------------------------------------------
def prepare_cifar10_data(cfg: Dict, pca_labels: bool = False) -> Tuple[Dict, Dict[str, DataLoader]]:
    base_path = cfg.get("dataset_path") or os.path.join("datasets", "obj_cls", "cifar10")
    splits = ["train", "test"]
    datasets_dict, loaders_dict = {}, {}
    
    for split in splits:
        transform = get_transform(
            ds_stats="cifar10",
            data_augment=(split == "train" and cfg.get("data_augment", True)),
            image_size=32
        )
        base_dataset = datasets.CIFAR10(
            base_path,
            train=(split == "train"),
            transform=transform,
            download=True
        )
        dataset = maybe_wrap_with_pca(base_dataset, base_path, cfg, split, pca_labels)
        datasets_dict[split] = dataset
        loaders_dict[split] = create_dataloader(
            dataset,
            batch_size=cfg.get("batchsize", 32),
            num_workers=cfg.get("num_workers", 8),
            shuffle=(split == "train")
        )
        print(f"CIFAR-10 {split} dataset: {len(dataset)} samples")
    return datasets_dict, loaders_dict


def prepare_tinyimgnet_data(cfg: Dict, pca_labels: bool = False) -> Tuple[Dict, Dict[str, DataLoader]]:
    base_path = cfg.get("dataset_path") or os.path.join("/data/shared/datasets", "tiny-imagenet")
    pca_base_path = os.path.join("datasets", "obj_cls", "tiny-imagenet")  # Path for PCA labels
    splits = ["train", "test"]
    datasets_dict, loaders_dict = {}, {}
    
    for split in splits:
        augment = cfg.get("data_augment", True) and split == "train"
        transform_list = [
            transforms.Resize(64),  # Tiny ImageNet is 64x64
            transforms.CenterCrop(64),  # Ensure consistent size
        ]
        
        # Add augmentation only for training
        if augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),  # 50% chance of flip
                transforms.RandomRotation(10),  # Reduced from 15 to 10 degrees
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)  # Reduced intensity, removed hue
            ])
            
        # Add normalization
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=DS_MEAN["tiny-imagenet"], std=DS_STD["tiny-imagenet"])
        ])
        
        transform = transforms.Compose(transform_list)
        
        base_dataset = TinyImageNetDataset(base_path, split, transform)
        # Set n_classes in config based on dataset if not using PCA labels
        if not pca_labels and "n_classes" not in cfg:
            cfg["n_classes"] = base_dataset.num_classes
            print(f"Setting n_classes to {base_dataset.num_classes} based on dataset")
        
        # Debug: Verify a batch of data
        test_loader = DataLoader(
            base_dataset,
            batch_size=min(8, cfg.get("batchsize", 32)),  # Use smaller batch for testing
            shuffle=False,
            num_workers=0  # Use single worker for debugging
        )
        images, labels = next(iter(test_loader))
        print(f"\nDebug batch for {split}:")
        print(f"Image shape: {images.shape}, dtype: {images.dtype}")
        print(f"Label shape: {labels.shape}, dtype: {labels.dtype}")
        print(f"Label range: [{labels.min().item()}, {labels.max().item()}]")
        print(f"Image range: [{images.min().item():.3f}, {images.max().item():.3f}]")
        print(f"Image mean: {images.mean().item():.3f}")
        print(f"Image std: {images.std().item():.3f}")
        
        dataset = maybe_wrap_with_pca(base_dataset, pca_base_path, cfg, split, pca_labels)
        datasets_dict[split] = dataset
        loaders_dict[split] = create_dataloader(
            dataset,
            batch_size=cfg.get("batchsize", 32),
            num_workers=cfg.get("num_workers", 8),
            shuffle=(split == "train")
        )
        print(f"Tiny ImageNet {split} dataset: {len(dataset)} samples")
    return datasets_dict, loaders_dict


def prepare_imgnet_data(cfg: Dict, pca_labels: bool = False) -> Tuple[Dict, Dict[str, DataLoader]]:
    base_path = cfg.get("dataset_path") or os.path.join("datasets", "obj_cls", "imagenet")
    splits = ["train", "test"]
    datasets_dict, loaders_dict = {}, {}
    
    for split in splits:
        transform = get_transform(
            ds_stats="imgnet",
            data_augment=(split == "train" and cfg.get("data_augment", True)),
            image_size=224
        )
        base_dataset = ImageNetDataset(
            base_path,
            split=split,
            transform=transform,
            train_ratio=cfg.get("train_ratio", 0.8)
        )
        dataset = maybe_wrap_with_pca(base_dataset, base_path, cfg, split, pca_labels)
        datasets_dict[split] = dataset
        loaders_dict[split] = create_dataloader(
            dataset,
            batch_size=cfg.get("batchsize", 32),
            num_workers=cfg.get("num_workers", 8),
            shuffle=(split == "train")
        )
        print(f"ImageNet {split} dataset: {len(dataset)} samples")
    return datasets_dict, loaders_dict


# -----------------------------------------------------------------------------
# Main loader function
# -----------------------------------------------------------------------------
def get_obj_cls_loader(cfg: Dict) -> Tuple[Dict, Dict[str, DataLoader]]:
    """
    Prepare object classification datasets and loaders.
    Supported datasets: 'tiny-imagenet', 'imagenet', and 'cifar10'.
    
    Args:
        cfg: Configuration dictionary.
        
    Returns:
        A tuple (datasets_dict, loaders_dict) mapping split names to the respective objects.
    """
    dataset_name = cfg.get("dataset", "tiny-imagenet")
    pca_labels = cfg.get("pca_labels", False)  # Get from config with default False
    
    if dataset_name == "tiny-imagenet":
        return prepare_tinyimgnet_data(cfg, pca_labels)
    elif dataset_name == "imagenet":
        return prepare_imgnet_data(cfg, pca_labels)
    elif dataset_name == "cifar10":
        return prepare_cifar10_data(cfg, pca_labels)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")