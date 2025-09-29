import os
import json
import torch
import warnings
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image
import pandas as pd

import visreps.utils as utils

# Filter out PIL TiffImagePlugin truncated file warnings
warnings.filterwarnings('ignore', category=UserWarning, module='PIL.TiffImagePlugin')

# Global normalization statistics.
DS_MEAN = {
    "tiny-imagenet": [0.480, 0.448, 0.398],
    "imgnet": [0.485, 0.456, 0.406],
}
DS_STD = {
    "tiny-imagenet": [0.272, 0.265, 0.274],
    "imgnet": [0.229, 0.224, 0.225],
}

def get_transform(ds_stats="imgnet", data_augment=False, image_size=224):
    """Return a composed transform based on dataset stats and augmentation flag."""
    if ds_stats == "tiny-imagenet":
        resize_size, crop_size = 64, 64
    else:
        resize_size, crop_size = 256, image_size

    tfms = [
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(crop_size)
    ]
    if data_augment:
        tfms += [transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)]
    tfms += [transforms.ToTensor(), transforms.Normalize(DS_MEAN[ds_stats], DS_STD[ds_stats])]
    return transforms.Compose(tfms)

# -----------------------------------------------------------------------------
# PCA Dataset wrapper
# -----------------------------------------------------------------------------
class PCADataset(Dataset):
    """
    Wraps a base dataset to substitute its labels with PCA-derived ones.
    Expects a CSV with 'image' and 'pca_label' columns.
    """
    def __init__(self, base_dataset, pca_labels_path, num_classes: int):
        self.dataset = base_dataset
        self.label_map = self._load_pca_labels(pca_labels_path)
        # Store the number of PCA classes
        self.num_classes = num_classes 
        self._filter_samples()

    def _load_pca_labels(self, csv_path):
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise RuntimeError(f"Error reading PCA CSV at {csv_path}: {e}")

        for col in ["image", "pca_label"]:
            if col not in df.columns:
                raise ValueError(f"PCA CSV must include '{col}'")
        if df["pca_label"].dtype.kind not in "iu" or df["pca_label"].min() < 0:
            raise ValueError("PCA labels must be non-negative integers")
        return {os.path.basename(row["image"]): int(row["pca_label"]) for _, row in df.iterrows()}

    def _filter_samples(self):
        """Filter samples to only those with PCA labels."""
        if not hasattr(self.dataset, "samples"):
            return
        
        filtered_samples = []
        for sample in self.dataset.samples:
            img_id = os.path.basename(sample[2])
            if img_id in self.label_map:
                filtered_samples.append(sample)
        
        total = len(self.dataset.samples)
        kept = len(filtered_samples)
        print(f"Filtered dataset from {total} to {kept} samples with PCA labels ({kept/total*100:.1f}%)")
        self.dataset.samples = filtered_samples

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        img_id = os.path.basename(self.dataset.samples[idx][2])
        label = self.label_map[img_id]
        return image, label

# -----------------------------------------------------------------------------
# Dataset classes
# -----------------------------------------------------------------------------
class ImageNetDataset(Dataset):
    """
    Custom loader for ImageNet with a flat folder structure.
    Folder-to-label mapping is read from a JSON file.
    Can load 'train', 'test', or 'all' splits.
    """
    def __init__(self, base_path, split = "train", transform=None, train_ratio= 0.8):
        assert split in ["train", "test", "all"], f"Invalid split: {split}"
        self.transform = transform
        label_file = os.path.join(utils.get_env_var("IMAGENET_LOCAL_DIR"), "folder_labels.json")
        self.num_classes = 1000
        
        # Load folder -> label mapping
        try:
            with open(label_file, "r") as f:
                self.folder_labels = json.load(f)
        except FileNotFoundError:
             raise FileNotFoundError(f"Label file not found: {label_file}")
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON from {label_file}")

        self.samples = []
        skipped = set()

        # Scan for all valid images first
        valid_folders = set(self.folder_labels.keys())
        if not os.path.isdir(base_path):
             raise FileNotFoundError(f"ImageNet base path not found or not a directory: {base_path}")
             
        for folder in os.listdir(base_path):
            if not folder.startswith("n"): # Standard ImageNet folder prefix
                continue
            folder_path = os.path.join(base_path, folder)
            # Check if folder is valid and exists in label file
            if not os.path.isdir(folder_path) or folder not in valid_folders:
                skipped.add(folder)
                continue
                
            label = int(self.folder_labels[folder])
            for fname in os.listdir(folder_path):
                # Check for standard image extensions
                if fname.lower().endswith((".jpeg", ".jpg")):
                    img_path = os.path.join(folder_path, fname)
                    img_id = fname  # Use filename for potential PCA matching later
                    self.samples.append((img_path, label, img_id))
                    
        total_found = len(self.samples)

        # Apply train/test split only if split is 'train' or 'test'
        if split in ["train", "test"]:
            if total_found == 0:
                 self.samples = [] # Ensure samples is empty list
            else:
                 # Use a fixed seed for reproducible splits if needed, otherwise random split
                 # torch.manual_seed(42) # Uncomment for deterministic split
                 indices = torch.randperm(total_found).tolist()
                 split_idx = int(total_found * train_ratio)
                 if split == "train":
                     self.samples = [self.samples[i] for i in indices[:split_idx]]
                 else: # split == "test"
                     self.samples = [self.samples[i] for i in indices[split_idx:]]
        # If split is 'all', self.samples remains the full list

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label, _ = self.samples[idx]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

class TinyImageNetDataset(Dataset):
    """
    Loader for Tiny ImageNet using torchvision's ImageFolder.
    Also prints dataset and transform pipeline info.
    """
    def __init__(self, base_path: str, split: str, transform=None):
        self.split_folder = "train" if split == "train" else "val"
        self.root = os.path.join(base_path, self.split_folder)
        self.dataset = datasets.ImageFolder(self.root, transform=transform)
        self.loader = self.dataset.loader
        self.transform = self.dataset.transform
        self.num_classes = len(self.dataset.classes)

        class_counts = {}
        for _, label in self.dataset.samples:
            class_counts[label] = class_counts.get(label, 0) + 1

        self.samples = [(path, label, os.path.relpath(path, self.root))
                        for path, label in self.dataset.samples]
        for path, label, _ in self.samples:
            if not (0 <= label < self.num_classes):
                raise ValueError(f"Invalid label {label} for image {path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label, _ = self.samples[idx]
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
            if idx < 5:
                if not (isinstance(image, torch.Tensor) and image.dim() == 3 and image.shape[0] == 3):
                    raise ValueError(f"Unexpected image shape {image.shape} for {path}")
        return image, label

# -----------------------------------------------------------------------------
# DataLoader helpers
# -----------------------------------------------------------------------------
def create_collate_fn():
    """Collate function for (image, label) pairs."""
    def collate_fn(batch):
        images, labels = zip(*batch)
        return torch.stack(images), torch.tensor(labels)
    return collate_fn

def create_dataloader(dataset: Dataset, batch_size: int = 32, num_workers: int = 4,
                      shuffle: bool = True, collate_fn=None) -> DataLoader:
    # Conditionally set prefetch_factor only if using multiple workers
    prefetch_factor = 8 if num_workers > 0 else None
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor, # Use the conditional value
        pin_memory=True,
        collate_fn=collate_fn or create_collate_fn()
    )

def wrap_with_pca(dataset, base_path, cfg, split):
    """Wrap the dataset with PCA labels"""
    n_classes = cfg.get("pca_n_classes")
    if n_classes is None:
        raise ValueError("pca_n_classes must be specified in config when pca_labels=True")
    pca_file = f"n_classes_{n_classes}.csv"
    pca_labels_path = os.path.join(base_path, pca_file)
    print(f"Applying PCA labels for {split} from {pca_labels_path}")
    return PCADataset(dataset, pca_labels_path, num_classes=n_classes)

# -----------------------------------------------------------------------------
# Dataset preparation functions
# -----------------------------------------------------------------------------
def prepare_tinyimgnet_data(cfg, pca_labels, shuffle):
    base_path = cfg.get("dataset_path", utils.get_env_var("TINY_IMAGENET_DATA_DIR"))

    # Fetch the local dir path first to trigger potential error from get_env_var
    local_dir_path = utils.get_env_var("TINY_IMAGENET_LOCAL_DIR")

    # PCA labels are stored in project root's pca_labels directory
    pca_base_path = os.path.join("pca_labels", cfg.get("pca_labels_folder"))

    datasets, loaders = {}, {}

    # Determine splits: Use 'val' as 'all' for extraction (shuffle=False), otherwise use ['train', 'test']
    # Tiny ImageNet conventionally uses 'val' for testing/evaluation.
    splits_to_load = ["val"] if not shuffle else ["train", "val"]
    split_info = []

    for split in splits_to_load:
        # Determine actual folder name ('train' or 'val')
        folder_split = "train" if split == "train" else "val"
        
        # Augmentation only for train split when shuffle=True
        augment = cfg.get("data_augment", True) and split == "train" and shuffle
        tfms = (
            [transforms.Resize(64), transforms.CenterCrop(64)]
            + ([
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(10),
                transforms.ColorJitter(0.2, 0.2, 0.2)
              ] if augment else [])
            + [transforms.ToTensor(), transforms.Normalize(DS_MEAN["tiny-imagenet"], DS_STD["tiny-imagenet"])]
        )
        transform = transforms.Compose(tfms)
        # Use the folder_split to point to the correct directory
        dataset = TinyImageNetDataset(base_path, folder_split, transform)

        # Use the main split name ('train', 'val', or potentially 'all' if we rename 'val') for PCA wrapping
        dataset = wrap_with_pca(dataset, pca_base_path, cfg, split) if pca_labels else dataset
        
        # Use the main split name ('train' or 'val') as the key in the returned dict
        # If shuffle is False, we want the key to be 'all', so we rename 'val' to 'all'
        dict_key = "all" if not shuffle and split == "val" else split
        datasets[dict_key] = dataset
        loaders[dict_key] = create_dataloader(
            dataset,
            batch_size=cfg.get("batchsize", 32),
            num_workers=cfg.get("num_workers", 4),
            shuffle=shuffle # Pass the original shuffle flag
        )
        split_info.append(f"{dict_key}={len(dataset)}")

    print(f"📊 Tiny ImageNet: {', '.join(split_info)}")
    return datasets, loaders

def prepare_imgnet_data(cfg, pca_labels, shuffle, base_path=None):
    """Prepares ImageNet or related datasets (like mini variants).

    Args:
        cfg: Configuration object.
        pca_labels: Boolean indicating if PCA labels should be used.
        shuffle: Boolean indicating if data should be shuffled.
        base_path: Direct path to dataset. If None, uses IMAGENET_DATA_DIR env var.
    """
    if base_path is None:
        base_path = cfg.get("dataset_path", utils.get_env_var("IMAGENET_DATA_DIR"))
    datasets, loaders = {}, {}

    # Determine splits: For feature extraction (shuffle=False), use 'all'. Otherwise, use ['train', 'test']
    splits_to_load = ["all"] if not shuffle else ["train", "test"]
    split_info = []

    for split in splits_to_load:
        # Augmentation is usually False during feature extraction (controlled by shuffle flag proxy)
        augment = cfg.get("data_augment", False) and split == "train" and shuffle
        tfms = get_transform(ds_stats="imgnet", data_augment=augment, image_size=224)
        
        # Instantiate the dataset for the current split ('train', 'test', or 'all')
        dataset = ImageNetDataset(base_path, split=split, transform=tfms)

        # Wrap with PCA labels if specified (usually only during training/evaluation, not extraction)
        if pca_labels:
            pca_base_path = os.path.join("pca_labels", cfg.get("pca_labels_folder"))
            dataset = wrap_with_pca(dataset, pca_base_path, cfg, split)
        
        datasets[split] = dataset
        loaders[split] = create_dataloader(
            dataset,
            batch_size=cfg.get("batchsize", 512),
            num_workers=cfg.get("num_workers", 8),
            shuffle=shuffle, # Shuffle should be False for extraction split='all'
        )
        split_info.append(f"{split}={len(dataset)}")

    print(f"📊 ImageNet: {', '.join(split_info)}")
    return datasets, loaders

def get_obj_cls_loader(cfg, shuffle=True):
    """Return datasets and dataloaders for object classification."""
    dataset_name = cfg.get("dataset", "tiny-imagenet")
    pca_labels = cfg.get("pca_labels", False)

    if dataset_name == "tiny-imagenet":
        datasets, loaders = prepare_tinyimgnet_data(cfg, pca_labels, shuffle)
    elif dataset_name == "imagenet":
        datasets, loaders = prepare_imgnet_data(cfg, pca_labels, shuffle)
    elif dataset_name.startswith("imagenet-mini-"):
        # Extract number of images per class from dataset name
        try:
            num_images = int(dataset_name.split("-")[-1])
        except ValueError:
            raise ValueError(f"Invalid imagenet-mini format: {dataset_name}. Expected imagenet-mini-<number>")
        
        # Construct path to mini dataset (sibling of main ImageNet)
        imagenet_base = Path(utils.get_env_var("IMAGENET_DATA_DIR"))
        mini_path = imagenet_base.parent / f"imagenet-mini-{num_images}"
        
        if not mini_path.exists():
            raise ValueError(f"ImageNet mini dataset not found at {mini_path}")
        
        datasets, loaders = prepare_imgnet_data(cfg, pca_labels, shuffle, base_path=str(mini_path))
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return datasets, loaders