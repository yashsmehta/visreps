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
    """
    transform_list = [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),
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
        # Store original paths for image names
        self.samples = []
        for path, label in self.dataset.samples:
            # Get relative path from dataset root for consistent naming
            rel_path = os.path.relpath(path, self.root)
            self.samples.append((path, label, rel_path))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int):
        path, label, _ = self.samples[idx]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
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
    The CSV file is expected at: <base_path>/pca_labels/n_classes_{2**n_bits}.csv.
    """
    if pca_labels:
        n_classes = cfg.get("n_classes", 4)
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
    splits = ["train", "test"]
    datasets_dict, loaders_dict = {}, {}
    
    for split in splits:
        augment = cfg.get("data_augment", True) and split == "train"
        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.RandomHorizontalFlip() if augment else transforms.Lambda(lambda x: x),
            transforms.RandomRotation(15) if augment else transforms.Lambda(lambda x: x),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1) if augment else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        
        base_dataset = TinyImageNetDataset(base_path, split, transform)
        dataset = maybe_wrap_with_pca(base_dataset, base_path, cfg, split, pca_labels)
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
def get_obj_cls_loader(cfg: Dict, pca_labels: bool = True) -> Tuple[Dict, Dict[str, DataLoader]]:
    """
    Prepare object classification datasets and loaders.
    Supported datasets: 'tiny-imagenet', 'imagenet', and 'cifar10'.
    
    Args:
        cfg: Configuration dictionary.
        pca_labels: Whether to replace original labels with PCA-derived ones.
        
    Returns:
        A tuple (datasets_dict, loaders_dict) mapping split names to the respective objects.
    """
    dataset_name = cfg.get("dataset", "tiny-imagenet")
    if dataset_name == "tiny-imagenet":
        return prepare_tinyimgnet_data(cfg, pca_labels)
    elif dataset_name == "imagenet":
        return prepare_imgnet_data(cfg, pca_labels)
    elif dataset_name == "cifar10":
        return prepare_cifar10_data(cfg, pca_labels)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")