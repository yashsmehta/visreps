import os
import json
import torch
import warnings
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd

import visreps.utils as utils

# Filter out PIL TiffImagePlugin truncated file warnings
warnings.filterwarnings('ignore', category=UserWarning, module='PIL.TiffImagePlugin')

# Global normalization statistics.
DS_MEAN = {
    "imgnet": [0.485, 0.456, 0.406],
    "clip":   [0.48145466, 0.4578275, 0.40821073],
}
DS_STD = {
    "imgnet": [0.229, 0.224, 0.225],
    "clip":   [0.26862954, 0.26130258, 0.27577711],
}

def get_transform(ds_stats="imgnet", data_augment=False, image_size=224, preprocess=True,
                   val_resize_size=256, augment_type="standard"):
    """Return a composed transform based on dataset stats and augmentation flag.

    ``augment_type`` controls the training augmentation strategy:
      - ``"standard"``: RandomResizedCrop + RandomHorizontalFlip (modern ImageNet recipe)
      - ``"mild"``: Resize + CenterCrop + RandomHorizontalFlip + RandomRotation(10)

    ``val_resize_size`` controls the resize dimension for validation transforms
    (default 256, but modern recipes like ResNet50-V2 / ConvNeXt use 232).
    """
    if not preprocess:
        return transforms.Compose([transforms.ToTensor()])

    if data_augment:
        if augment_type == "mild":
            # Mild: whole-object view with light perturbation
            tfms = [
                transforms.Resize(val_resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
            ]
        else:
            # Standard: RandomResizedCrop + RandomHorizontalFlip
            tfms = [
                transforms.RandomResizedCrop(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.RandomHorizontalFlip(),
            ]
    else:
        # Validation / test: deterministic Resize + CenterCrop
        tfms = [
            transforms.Resize(val_resize_size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(image_size),
        ]

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
    def __init__(self, base_path, split = "train", transform=None, train_ratio= 0.8, train_fraction=1.0):
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
                    
        # Sort for filesystem-independent ordering
        self.samples.sort(key=lambda s: s[2])
        total_found = len(self.samples)

        # Apply train/test split only if split is 'train' or 'test'
        if split in ["train", "test"]:
            if total_found == 0:
                 self.samples = []
            elif total_found <= self.num_classes:
                 # Too few images per class to split meaningfully — use all for both
                 print(f"⚠️ Only {total_found} images for {self.num_classes} classes — using all for both train and test")
            else:
                 g = torch.Generator().manual_seed(42)
                 indices = torch.randperm(total_found, generator=g).tolist()
                 split_idx = int(total_found * train_ratio)
                 if split == "train":
                     self.samples = [self.samples[i] for i in indices[:split_idx]]
                 else: # split == "test"
                     self.samples = [self.samples[i] for i in indices[split_idx:]]
        # If split is 'all', self.samples remains the full list

        # Subsample training split if train_fraction < 1.0
        if split == "train" and train_fraction < 1.0 and len(self.samples) > 0:
            g = torch.Generator().manual_seed(42)
            n_keep = max(1, int(len(self.samples) * train_fraction))
            indices = torch.randperm(len(self.samples), generator=g).tolist()[:n_keep]
            self.samples = [self.samples[i] for i in sorted(indices)]
            print(f"train_fraction={train_fraction}: kept {n_keep} of {split_idx} train samples")

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

    def get_wnid_from_label(self, label_idx: int) -> str:
        """Convert a class index (0-999) to its WordNet ID."""
        for wnid, idx in self.folder_labels.items():
            if int(idx) == label_idx:
                return wnid
        raise ValueError(f"Label index {label_idx} not found.")

    def get_wordnet_synset(self, label_idx: int):
        """Returns the NLTK Synset object for the class index."""
        import nltk
        from nltk.corpus import wordnet as wn
        
        try: wn.ensure_loaded()
        except LookupError: nltk.download('wordnet'); nltk.download('omw-1.4')

        wnid = self.get_wnid_from_label(label_idx)
        try:
            return wn.synset_from_pos_and_offset('n', int(wnid[1:]))
        except Exception as e:
            print(f"Error retrieving synset for {wnid}: {e}")
            return None

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
def _resolve_dataset_path(cfg):
    """Resolve the dataset base path from config or environment."""
    dataset_name = cfg.get("dataset", "imagenet")
    if dataset_name.startswith("imagenet-mini-"):
        try:
            num_images = int(dataset_name.split("-")[-1])
        except ValueError:
            raise ValueError(f"Invalid imagenet-mini format: {dataset_name}. Expected imagenet-mini-<number>")
        mini_path = Path(utils.get_env_var("IMAGENET_DATA_DIR")).parent / f"imagenet-mini-{num_images}"
        if not mini_path.exists():
            raise ValueError(f"ImageNet mini dataset not found at {mini_path}")
        return str(mini_path)
    return cfg.get("dataset_path", utils.get_env_var("IMAGENET_DATA_DIR"))

def prepare_imgnet_data(cfg, pca_labels, shuffle, preprocess, train_test_split):
    """Prepares ImageNet or ImageNet-mini datasets."""
    base_path = _resolve_dataset_path(cfg)
    datasets, loaders = {}, {}

    # Determine splits based on train_test_split flag
    splits_to_load = ["train", "test"] if train_test_split else ["all"]
    split_info = []

    for split in splits_to_load:
        augment = cfg.get("data_augment", False) and split == "train" and shuffle and preprocess
        augment_type = cfg.get("augment_type", "standard")
        tfms = get_transform(ds_stats="imgnet", data_augment=augment, image_size=224,
                             preprocess=preprocess, augment_type=augment_type)
        
        # Instantiate the dataset for the current split ('train', 'test', or 'all')
        train_fraction = cfg.get("train_fraction", 1.0)
        dataset = ImageNetDataset(base_path, split=split, transform=tfms, train_fraction=train_fraction)

        # Wrap with PCA labels if specified
        if pca_labels:
            pca_base_path = os.path.join("pca_labels", cfg.get("pca_labels_folder"))
            dataset = wrap_with_pca(dataset, pca_base_path, cfg, split)
        
        datasets[split] = dataset
        loaders[split] = create_dataloader(
            dataset,
            batch_size=cfg.get("batchsize", 512),
            num_workers=cfg.get("num_workers", 8),
            shuffle=shuffle,
        )
        split_info.append(f"{split}={len(dataset)}")

    print(f"📊 ImageNet: {', '.join(split_info)}")
    return datasets, loaders

def get_obj_cls_loader(cfg, shuffle=True, preprocess=True, train_test_split=True):
    """Return datasets and dataloaders for object classification."""
    dataset_name = cfg.get("dataset", "imagenet")
    if not (dataset_name == "imagenet" or dataset_name.startswith("imagenet-mini-")):
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    pca_labels = cfg.get("pca_labels", False)
    return prepare_imgnet_data(cfg, pca_labels, shuffle, preprocess, train_test_split)
