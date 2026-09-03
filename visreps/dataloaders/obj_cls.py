"""ImageNet dataloader (parquet-backed) for visreps training.

This module wraps ``imagenet_loader.ImageNetParquet`` (parquet shards on
``/datadisk``) in a thin compatibility shim that preserves the historical
public API (``get_transform``, ``get_obj_cls_loader`` returning
``(datasets, loaders)`` keyed by split).

Notes on this rewrite (new machine, 2026):
    * The legacy folder-based ``ImageNetDataset`` (Bonner-lab layout with
      ``folder_labels.json`` and a 80/20 random split) is kept at
      ``obj_cls.py.legacy`` for reference.
    * Train/test mapping is now the official HuggingFace
      ``train`` / ``validation`` splits — there is a *one-time* discontinuity
      in reported test accuracy versus historical runs.
    * If ``imagenet_loader`` is not installed (lab cluster), everything is
      delegated to the folder backend in ``obj_cls_folder.py``, which also
      supports coarse labels (``pca_labels=True``). The parquet path itself
      does not support coarse labels yet.
    * ``imagenet-mini-{50, 100, 200}`` are no longer available.
"""

from __future__ import annotations

import math
import random
import warnings
from typing import Callable, Optional, Tuple

import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset
import torchvision.transforms as transforms
from torchvision.transforms.v2 import functional as F_v2

try:
    from imagenet_loader import (
        Collate,
        ImageNetParquet,
        eval_transform as _eval_transform_uint8,
        list_shards,
        train_transform as _train_transform_uint8,
    )
except ModuleNotFoundError:
    # Neural evaluation only needs ``get_transform`` below. Keep that path
    # usable on machines where the optional parquet training loader is absent.
    Collate = ImageNetParquet = list_shards = None
    _eval_transform_uint8 = _train_transform_uint8 = None

# ---------------------------------------------------------------------------
# PIL-pipeline transform helper (kept for evals.py / neural.py consumers)
# ---------------------------------------------------------------------------
DS_MEAN = {
    "imgnet": [0.485, 0.456, 0.406],
    "clip":   [0.48145466, 0.4578275, 0.40821073],
}
DS_STD = {
    "imgnet": [0.229, 0.224, 0.225],
    "clip":   [0.26862954, 0.26130258, 0.27577711],
}


def get_transform(
    ds_stats: str = "imgnet",
    data_augment: bool = False,
    image_size: int = 224,
    preprocess: bool = True,
    val_resize_size: int = 256,
    augment_type: str = "standard",
):
    """Return a torchvision PIL-pipeline transform.

    Used by ``evals.py`` and ``dataloaders/neural.py`` for non-training image
    preprocessing (PIL → tensor). The training pipeline does **not** go
    through this function — see ``_make_train_transform`` /
    ``_make_eval_transform`` for the uint8 CHW transforms applied inside
    the parquet loader.
    """
    if not preprocess:
        return transforms.Compose([transforms.ToTensor()])

    if data_augment:
        if augment_type == "mild":
            tfms = [
                transforms.Resize(
                    val_resize_size,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.CenterCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
            ]
        else:
            tfms = [
                transforms.RandomResizedCrop(
                    image_size,
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.RandomHorizontalFlip(),
            ]
    else:
        tfms = [
            transforms.Resize(
                val_resize_size,
                interpolation=transforms.InterpolationMode.BILINEAR,
            ),
            transforms.CenterCrop(image_size),
        ]

    tfms += [
        transforms.ToTensor(),
        transforms.Normalize(DS_MEAN[ds_stats], DS_STD[ds_stats]),
    ]
    return transforms.Compose(tfms)


# ---------------------------------------------------------------------------
# uint8 CHW transforms (operate inside the parquet pipeline, before Collate)
# ---------------------------------------------------------------------------
def _eval_transform(image_size: int = 224):
    """Resize + center-crop on uint8 CHW (matches ``imagenet_loader``)."""

    def _fn(img: torch.Tensor) -> torch.Tensor:
        return _eval_transform_uint8(img, image_size=image_size)

    return _fn


def _train_transform_standard(image_size: int = 224):
    """RandomResizedCrop + HFlip on uint8 CHW (matches ``imagenet_loader``)."""

    def _fn(img: torch.Tensor) -> torch.Tensor:
        return _train_transform_uint8(img, image_size=image_size)

    return _fn


def _train_transform_mild(image_size: int = 224, val_resize_size: int = 256):
    """Mild augmentation on uint8 CHW: Resize + CenterCrop + HFlip + Rotate(±10°).

    Mirrors the legacy ``augment_type='mild'`` pipeline used for
    ``custom_model``, but operates on uint8 CHW tensors so it composes with
    ``Collate``'s fused normalize step.
    """

    def _fn(img: torch.Tensor) -> torch.Tensor:
        img = F_v2.resize(img, val_resize_size, antialias=True)
        img = F_v2.center_crop(img, image_size)
        if random.random() < 0.5:
            img = F_v2.hflip(img)
        angle = random.uniform(-10.0, 10.0)
        img = F_v2.rotate(img, angle)
        return img

    return _fn


# ---------------------------------------------------------------------------
# Sized parquet wrapper — adds __len__ and num_classes for trainer compat
# ---------------------------------------------------------------------------
class _SizedImageNetParquet(IterableDataset):
    """``ImageNetParquet`` + ``__len__`` + per-iteration epoch bump.

    The visreps trainer calls ``len(loader)`` to compute average loss
    (``trainer.train_epoch``) and reads ``datasets['train'].num_classes``.
    PyTorch's ``DataLoader.__len__`` is defined for ``IterableDataset`` only
    if the dataset itself implements ``__len__``.

    Per-iteration epoch bump: the trainer never calls ``set_epoch``, but
    we still want each epoch to produce a fresh shard order. We bump the
    underlying ``_epoch`` counter on every ``__iter__`` call so each pass
    gets a different shuffle seed.
    """

    num_classes: int = 1000

    def __init__(
        self,
        shards,
        *,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        shuffle_shards: bool = False,
        seed: int = 0,
        keep_predicate: Optional[Callable[[str, int, int], bool]] = None,
    ):
        super().__init__()
        if not shards:
            raise ValueError("shards must be non-empty")
        self._inner = ImageNetParquet(
            shards,
            transform=transform,
            shuffle_shards=shuffle_shards,
            seed=seed,
        )
        self._keep_predicate = keep_predicate
        self._iter_count = 0
        self._length = self._compute_length(shards, keep_predicate)

    @staticmethod
    def _compute_length(shards, keep_predicate) -> int:
        if keep_predicate is None:
            return sum(pq.ParquetFile(s).metadata.num_rows for s in shards)
        # keep_predicate filters at iteration time — count rows by scanning
        # parquet metadata + path/label columns. Worth it because trainer
        # uses len() for loss averaging only.
        n = 0
        for shard in shards:
            pf = pq.ParquetFile(shard)
            for batch in pf.iter_batches(batch_size=4096, columns=["image", "label"]):
                paths = batch.column("image").field("path").to_pylist()
                labels = batch.column("label").to_pylist()
                for idx, (p, lbl) in enumerate(zip(paths, labels)):
                    if keep_predicate(p, int(lbl), idx):
                        n += 1
        return n

    def set_epoch(self, epoch: int) -> None:
        self._inner.set_epoch(epoch)

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        self._iter_count += 1
        self._inner.set_epoch(self._iter_count)
        if self._keep_predicate is None:
            yield from self._inner
            return
        # Filtered iteration: re-decode shards but apply predicate before
        # the underlying transform/decode loop. We mirror the inner loop
        # to avoid decoding rows we'll throw away.
        from torch.utils.data import get_worker_info
        from torchvision.io import ImageReadMode, decode_image

        info = get_worker_info()
        shards = list(self._inner.shards)
        if self._inner.shuffle_shards:
            wid = info.id if info else 0
            random.Random(
                self._inner.seed + self._inner._epoch * 1009 + wid
            ).shuffle(shards)
        if info is not None:
            shards = shards[info.id :: info.num_workers]

        tfm = self._inner.transform
        for path in shards:
            for batch in pq.ParquetFile(path).iter_batches(
                batch_size=self._inner.row_group_batch,
                columns=["image", "label"],
            ):
                imgs = batch.column("image")
                paths = imgs.field("path").to_pylist()
                bs = imgs.field("bytes").to_pylist()
                ls = batch.column("label").to_pylist()
                for idx, (p, b, lbl) in enumerate(zip(paths, bs, ls)):
                    if not self._keep_predicate(p, int(lbl), idx):
                        continue
                    img = decode_image(
                        torch.frombuffer(bytearray(b), dtype=torch.uint8),
                        mode=ImageReadMode.RGB,
                    )
                    yield (tfm(img) if tfm else img), int(lbl)


# ---------------------------------------------------------------------------
# DataLoader helpers (kept for back-compat with neural.py)
# ---------------------------------------------------------------------------
def create_collate_fn():
    """Plain stack collate used by neural-data loaders.

    Note: the parquet training pipeline uses ``Collate()`` from
    ``imagenet_loader`` (uint8 → fused normalize). This helper is unrelated
    and only feeds neural-stimuli loaders that already produce normalized
    float tensors via ``get_transform``.
    """

    def collate_fn(batch):
        images, labels = zip(*batch)
        return torch.stack(images), torch.tensor(labels)

    return collate_fn


def create_dataloader(
    dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    collate_fn=None,
) -> DataLoader:
    prefetch_factor = 8 if num_workers > 0 else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=True,
        collate_fn=collate_fn or create_collate_fn(),
    )


# ---------------------------------------------------------------------------
# mini-10 deterministic 80/20 row split
# ---------------------------------------------------------------------------
def _mini10_split_predicate(split: str) -> Callable[[str, int, int], bool]:
    """Deterministic ~80/20 row split keyed on parquet ``image.path``."""
    assert split in {"train", "test"}
    keep_train = split == "train"

    def _pred(path: str, label: int, idx: int) -> bool:
        # Stable hash on path: 32-bit FNV-1a (avoids ``hash()`` randomness).
        h = 0x811C9DC5
        for byte in path.encode("utf-8"):
            h = ((h ^ byte) * 0x01000193) & 0xFFFFFFFF
        is_train = (h % 5) != 0  # ~80% train, ~20% test
        return is_train if keep_train else not is_train

    return _pred


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
_HF_SPLIT = {"train": "train", "test": "validation", "all": "train"}
_DATASET_MAP = {"imagenet": "full", "imagenet-mini-10": "mini-10"}


def _build_transform(cfg, split: str, *, shuffle: bool, preprocess: bool):
    """Return the uint8 CHW transform applied inside the parquet loader.

    ``preprocess=False`` returns ``None`` so the loader yields raw decoded
    uint8 CHW tensors and ``Collate(normalize=False)`` keeps them uint8.
    """
    if not preprocess:
        return None

    image_size = int(cfg.get("image_size", 224))
    augment = cfg.get("data_augment", False) and split == "train" and shuffle
    if not augment:
        return _eval_transform(image_size=image_size)

    augment_type = "mild" if cfg.get("model_class") == "custom_model" else "standard"
    if augment_type == "mild":
        return _train_transform_mild(image_size=image_size)
    return _train_transform_standard(image_size=image_size)


def prepare_imgnet_data(cfg, pca_labels, shuffle, preprocess, train_test_split):
    """Build parquet-backed ImageNet datasets + dataloaders."""
    if ImageNetParquet is None:
        # Lab cluster: no parquet package, ImageNet is a folder per class.
        from visreps.dataloaders.obj_cls_folder import prepare_imgnet_data as _folder
        return _folder(cfg, pca_labels, shuffle, preprocess, train_test_split)

    if pca_labels:
        raise NotImplementedError(
            "Coarse-label training is not yet wired up for the parquet loader; "
            "the label CSVs key on filenames the parquet `image.path` field "
            "doesn't expose. Use the folder backend (obj_cls_folder.py)."
        )

    dataset_name = cfg.get("dataset", "imagenet")
    if dataset_name not in _DATASET_MAP:
        raise ValueError(
            f"Unsupported dataset: {dataset_name!r}. Expected one of "
            f"{sorted(_DATASET_MAP)}."
        )
    loader_dataset = _DATASET_MAP[dataset_name]
    is_mini10 = loader_dataset == "mini-10"

    if cfg.get("train_fraction", 1.0) != 1.0:
        warnings.warn(
            "train_fraction != 1.0 is not supported by the parquet pipeline "
            "yet; ignoring.",
            stacklevel=2,
        )

    splits_to_load = ["train", "test"] if train_test_split else ["all"]

    datasets, loaders = {}, {}
    info_parts = []

    for split in splits_to_load:
        hf_split = _HF_SPLIT[split]

        if is_mini10 and split == "test":
            # mini-10 only ships HF train shards. Fall back to a deterministic
            # 80/20 row split keyed on `image.path` so train ≠ test.
            shards = list_shards("train", dataset=loader_dataset)
            keep_predicate = _mini10_split_predicate("test")
            warnings.warn(
                "imagenet-mini-10 has no validation shards — using a "
                "deterministic 80/20 hash split of the train shards for "
                "the 'test' split. Use the full ImageNet for real "
                "evaluation.",
                stacklevel=2,
            )
        elif is_mini10 and split == "train":
            shards = list_shards("train", dataset=loader_dataset)
            keep_predicate = _mini10_split_predicate("train")
        else:
            shards = list_shards(hf_split, dataset=loader_dataset)
            keep_predicate = None

        tfm = _build_transform(cfg, split, shuffle=shuffle, preprocess=preprocess)

        ds = _SizedImageNetParquet(
            shards,
            transform=tfm,
            shuffle_shards=(shuffle and split == "train"),
            seed=int(cfg.get("seed", 0)),
            keep_predicate=keep_predicate,
        )

        batch_size = int(cfg.get("batchsize", 256))
        num_workers = int(cfg.get("num_workers", 8))
        # Cap workers at the shard count (mini-10 has 1 shard).
        num_workers = max(0, min(num_workers, len(shards)))
        # Disable persistent_workers so per-iter set_epoch() updates reach
        # workers (workers are recreated each epoch).
        loaders[split] = DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False,
            prefetch_factor=2 if num_workers > 0 else None,
            collate_fn=Collate(normalize=preprocess),
        )
        datasets[split] = ds
        info_parts.append(f"{split}={len(ds)}")

    print(f"📊 ImageNet ({dataset_name}): {', '.join(info_parts)}")
    return datasets, loaders


def get_obj_cls_loader(cfg, shuffle: bool = True, preprocess: bool = True,
                       train_test_split: bool = True):
    """Return ``(datasets, loaders)`` for object classification training.

    See module docstring for the supported dataset names and split semantics.
    """
    pca_labels = cfg.get("pca_labels", False)
    return prepare_imgnet_data(cfg, pca_labels, shuffle, preprocess, train_test_split)
