"""Folder-based ImageNet backend (lab cluster).

Used by ``obj_cls.get_obj_cls_loader`` whenever the parquet ``imagenet_loader``
package is not installed. ImageNet lives as one folder per wnid under
``IMAGENET_DATA_DIR`` and ``imagenet-mini-<n>`` subsets live as sibling folders.

Coarse labels (``pca_labels=True``) come from
``pca_labels/{pca_labels_folder}/n_classes_{pca_n_classes}.csv`` with columns
``image`` (filename) and ``pca_label``. This covers PCA labels, WordNet labels and
the hand-made semantic labels alike; images missing from the CSV are dropped.
"""

import json
import os
import warnings
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

import visreps.utils as utils

warnings.filterwarnings("ignore", category=UserWarning, module="PIL.TiffImagePlugin")


class ImageNetDataset(Dataset):
    """ImageNet from a flat folder-per-class layout with a fixed 80/20 split."""

    def __init__(self, base_path, split="train", transform=None, train_ratio=0.8, train_fraction=1.0):
        assert split in ["train", "test", "all"], f"Invalid split: {split}"
        self.transform = transform
        self.num_classes = 1000
        label_file = os.path.join(utils.get_env_var("IMAGENET_LOCAL_DIR"), "folder_labels.json")
        with open(label_file) as f:
            self.folder_labels = json.load(f)

        if not os.path.isdir(base_path):
            raise FileNotFoundError(f"ImageNet base path not found: {base_path}")

        self.samples = []  # (img_path, label, img_id)
        for folder in os.listdir(base_path):
            folder_path = os.path.join(base_path, folder)
            if folder not in self.folder_labels or not os.path.isdir(folder_path):
                continue
            label = int(self.folder_labels[folder])
            for fname in os.listdir(folder_path):
                if fname.lower().endswith((".jpeg", ".jpg")):
                    self.samples.append((os.path.join(folder_path, fname), label, fname))
        self.samples.sort(key=lambda s: s[2])

        if split in ["train", "test"] and len(self.samples) > self.num_classes:
            g = torch.Generator().manual_seed(42)
            indices = torch.randperm(len(self.samples), generator=g).tolist()
            split_idx = int(len(self.samples) * train_ratio)
            keep = indices[:split_idx] if split == "train" else indices[split_idx:]
            self.samples = [self.samples[i] for i in keep]

        if split == "train" and train_fraction < 1.0 and self.samples:
            g = torch.Generator().manual_seed(42)
            n_keep = max(1, int(len(self.samples) * train_fraction))
            indices = torch.randperm(len(self.samples), generator=g).tolist()[:n_keep]
            self.samples = [self.samples[i] for i in sorted(indices)]
            print(f"train_fraction={train_fraction}: kept {n_keep} train samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, _ = self.samples[idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_wnid_from_label(self, label_idx):
        for wnid, idx in self.folder_labels.items():
            if int(idx) == label_idx:
                return wnid
        raise ValueError(f"Label index {label_idx} not found.")

    def get_wordnet_synset(self, label_idx):
        import nltk
        from nltk.corpus import wordnet as wn

        try:
            wn.ensure_loaded()
        except LookupError:
            nltk.download("wordnet")
            nltk.download("omw-1.4")
        wnid = self.get_wnid_from_label(label_idx)
        return wn.synset_from_pos_and_offset("n", int(wnid[1:]))


class CoarseLabelDataset(Dataset):
    """Replace ImageNet labels with coarse labels read from a filename→label CSV."""

    def __init__(self, base_dataset, csv_path, num_classes):
        df = pd.read_csv(csv_path)
        for col in ["image", "pca_label"]:
            if col not in df.columns:
                raise ValueError(f"Label CSV must include '{col}': {csv_path}")
        if df["pca_label"].min() < 0 or df["pca_label"].max() >= num_classes:
            raise ValueError(f"Labels in {csv_path} fall outside [0, {num_classes})")
        self.label_map = dict(zip(df["image"].map(os.path.basename), df["pca_label"].astype(int)))
        self.num_classes = num_classes
        self.dataset = base_dataset

        total = len(base_dataset.samples)
        base_dataset.samples = [s for s in base_dataset.samples if s[2] in self.label_map]
        print(f"Coarse labels: kept {len(base_dataset.samples)}/{total} images ({csv_path})")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image, self.label_map[self.dataset.samples[idx][2]]


def resolve_dataset_path(cfg):
    dataset_name = cfg.get("dataset", "imagenet")
    if dataset_name.startswith("imagenet-mini-"):
        mini_path = Path(utils.get_env_var("IMAGENET_DATA_DIR")).parent / dataset_name
        if not mini_path.exists():
            raise ValueError(f"ImageNet mini dataset not found at {mini_path}")
        return str(mini_path)
    return cfg.get("dataset_path", utils.get_env_var("IMAGENET_DATA_DIR"))


def prepare_imgnet_data(cfg, pca_labels, shuffle, preprocess, train_test_split):
    """Build folder-backed ImageNet datasets + dataloaders."""
    from visreps.dataloaders.obj_cls import create_dataloader, get_transform

    base_path = resolve_dataset_path(cfg)
    splits = ["train", "test"] if train_test_split else ["all"]
    datasets, loaders, info = {}, {}, []

    for split in splits:
        augment = cfg.get("data_augment", False) and split == "train" and shuffle and preprocess
        augment_type = "mild" if cfg.get("model_class") == "custom_model" else "standard"
        tfms = get_transform(ds_stats="imgnet", data_augment=augment, image_size=224,
                             preprocess=preprocess, augment_type=augment_type)
        dataset = ImageNetDataset(base_path, split=split, transform=tfms,
                                  train_fraction=cfg.get("train_fraction", 1.0))

        if pca_labels:
            n_classes = cfg.get("pca_n_classes")
            if n_classes is None:
                raise ValueError("pca_n_classes must be set when pca_labels=True")
            csv_path = os.path.join("pca_labels", cfg.get("pca_labels_folder"), f"n_classes_{n_classes}.csv")
            dataset = CoarseLabelDataset(dataset, csv_path, num_classes=int(n_classes))

        datasets[split] = dataset
        loaders[split] = create_dataloader(dataset, batch_size=cfg.get("batchsize", 512),
                                           num_workers=cfg.get("num_workers", 8), shuffle=shuffle)
        info.append(f"{split}={len(dataset)}")

    print(f"📊 ImageNet ({cfg.get('dataset', 'imagenet')}, folder backend): {', '.join(info)}")
    return datasets, loaders
