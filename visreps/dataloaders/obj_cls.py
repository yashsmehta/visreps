import os
import json
import torch
import warnings
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image
import pandas as pd

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
    def __init__(self, base_dataset, pca_labels_path):
        self.dataset = base_dataset
        self.label_map = self._load_pca_labels(pca_labels_path)
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
    """
    def __init__(self, base_path, split = "train", transform=None, train_ratio= 0.8):
        self.transform = transform
        label_file = os.path.join("datasets", "obj_cls", "imagenet", "folder_labels.json")
        self.num_classes = 1000
        with open(label_file, "r") as f:
            self.folder_labels = json.load(f)

        self.samples = []
        skipped = set()
        for folder in os.listdir(base_path):
            if not folder.startswith("n"):
                continue
            folder_path = os.path.join(base_path, folder)
            if not os.path.isdir(folder_path) or folder not in self.folder_labels:
                skipped.add(folder)
                continue
            label = int(self.folder_labels[folder])
            for fname in os.listdir(folder_path):
                if fname.lower().endswith((".jpeg", ".jpg")):
                    img_path = os.path.join(folder_path, fname)
                    img_id = fname  # Just use filename for PCA matching
                    self.samples.append((img_path, label, img_id))
                    
        if skipped:
            print(f"Warning: Skipped {len(skipped)} folders not in folder labels")
        
        total = len(self.samples)
        indices = torch.randperm(total).tolist()
        split_idx = int(total * train_ratio)
        self.samples = [self.samples[i] for i in (indices[:split_idx] if split == "train" else indices[split_idx:])]
        print(f"{split.capitalize()} split: {len(self.samples)} samples")

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

def create_dataloader(dataset: Dataset, batch_size: int = 32, num_workers: int = 8,
                      shuffle: bool = True, collate_fn=None) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=2,
        pin_memory=True,
        collate_fn=collate_fn or create_collate_fn()
    )

def wrap_with_pca(dataset, base_path, cfg, split):
    """Wrap the dataset with PCA labels"""
    n_classes = cfg.get("pca_n_classes")
    pca_file = f"n_classes_{n_classes}.csv"
    pca_path = os.path.join(base_path, "pca_labels", pca_file)
    if not os.path.exists(pca_path):
        print(f"Warning: PCA file not found at {pca_path}. Using original labels.")
        return dataset
    print(f"Applying PCA labels for {split} from {pca_path}")
    return PCADataset(dataset, pca_path)

# -----------------------------------------------------------------------------
# Dataset preparation functions
# -----------------------------------------------------------------------------
def prepare_tinyimgnet_data(cfg, pca_labels, shuffle):
    base_path = cfg.get("dataset_path", os.path.join("/data/shared/datasets", "tiny-imagenet"))
    pca_base_path = os.path.join("datasets", "obj_cls", "tiny-imagenet")
    datasets, loaders = {}, {}
    for split in ["train", "test"]:
        augment = cfg.get("data_augment", True) and split == "train"
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
        dataset = TinyImageNetDataset(base_path, split, transform)
        if not pca_labels and "n_classes" not in cfg:
            cfg["n_classes"] = dataset.num_classes
            print(f"Setting n_classes to {dataset.num_classes} based on dataset")
        dataset = wrap_with_pca(dataset, pca_base_path, cfg, split) if pca_labels else dataset
        datasets[split] = dataset
        loaders[split] = create_dataloader(
            dataset,
            batch_size=cfg.get("batchsize", 32),
            num_workers=cfg.get("num_workers", 8),
            shuffle=shuffle
        )
    return datasets, loaders

def prepare_imgnet_data(cfg, pca_labels, shuffle):
    base_path = cfg.get("dataset_path", "/data/shared/datasets/imagenet")
    pca_base_path = os.path.join("datasets", "obj_cls", "imagenet")
    datasets_dict, loaders_dict = {}, {}

    for split in ["train", "test"]:
        augment = split == "train" and cfg.get("data_augment", True)
        transform = get_transform(ds_stats="imgnet", data_augment=augment, image_size=224)
        dataset = ImageNetDataset(base_path, split=split, transform=transform, train_ratio=cfg.get("train_ratio", 0.8))
        
        if not pca_labels and "n_classes" not in cfg:
            cfg["n_classes"] = dataset.num_classes
            print(f"Setting n_classes to {dataset.num_classes} based on dataset")
        
        dataset = wrap_with_pca(dataset, pca_base_path, cfg, split) if pca_labels else dataset

        datasets_dict[split] = dataset
        loaders_dict[split] = create_dataloader(
            dataset,
            batch_size=cfg.get("batchsize", 32),
            num_workers=cfg.get("num_workers", 8),
            shuffle=shuffle
        )

    return datasets_dict, loaders_dict

def get_obj_cls_loader(cfg, shuffle=True):
    """
    Prepares object classification datasets and loaders.
    Supported datasets: 'tiny-imagenet' and 'imagenet'.
    """
    dataset_name = cfg.get("dataset", "tiny-imagenet")
    pca_labels = cfg.get("pca_labels", False)
    if dataset_name == "tiny-imagenet":
        return prepare_tinyimgnet_data(cfg, pca_labels, shuffle)
    elif dataset_name == "imagenet":
        return prepare_imgnet_data(cfg, pca_labels, shuffle)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")