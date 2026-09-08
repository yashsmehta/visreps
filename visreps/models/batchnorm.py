"""BatchNorm statistics for evaluation.

``bn_calibration_source`` selects where the running statistics come from:

* ``dataset`` (default): recalibrate on the neural dataset's training images
  (THINGS selection split; NSD/TVSD train images minus every subject's test set).
* ``imagenet``: recalibrate on a fixed random subset of 512,000 ImageNet training
  images in batches of 256, i.e. the distribution the weights were trained on.
  Use this for BatchNorm architectures (ResNet-50) whose checkpoint statistics are
  unreliable — see ``experiments/bn_recalibration/README.md``. The recipe is
  fixed: 128,000 images or batches of 64 under-sample the heavy-tailed channels
  of coarse-trained ResNet-50s (16-class THINGS RSA 0.48 instead of 0.56).
* ``checkpoint``: keep the checkpoint's own running statistics.

Only BN buffers are ever changed; weights stay fixed and the model is returned in
eval mode. Recalibrated buffers are cached in ``model_checkpoints/bn_stats/``.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

SOURCES = ("dataset", "imagenet", "checkpoint")
# Fixed recipes, not configurable: the calibration identity is part of every run_id.
IMAGENET_IMAGES = 512_000      # 2,000 batches of 256, the validated March 2026 recipe
IMAGENET_BATCH_SIZE = 256
DATASET_BATCH_SIZE = 64        # neural-dataset calibration (unchanged since 2026-09-05)
_REMOVED_OPTIONS = ("bn_calibration_images", "bn_calibration_batchsize")


def training_image_ids(neural, available):
    """Union of training images, excluding every participating subject's test set."""
    train, test = set(), set()
    for subjects in neural.values():
        for splits in subjects.values():
            train.update(splits["train"])
            test.update(splits["test"])
    return sorted((train - test) & set(available))


def _norm_layers(model):
    return {name: m for name, m in model.named_modules()
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))}


def _model_digest(model):
    """Hash the model, including buffers, so seeds/epochs cannot collide."""
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        value = value.detach().cpu().contiguous()
        digest.update(str((name, str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _reject_removed_options(cfg):
    """The calibration recipes are fixed; refuse configs that try to change them."""
    present = [k for k in _REMOVED_OPTIONS if cfg.get(k) is not None]
    if present:
        raise ValueError(f"{present} are not configurable: BatchNorm calibration uses "
                         f"{IMAGENET_IMAGES} ImageNet images in batches of {IMAGENET_BATCH_SIZE} "
                         f"(imagenet) or batches of {DATASET_BATCH_SIZE} (dataset).")


def _batches(n, batch_size, seed=0):
    """Deterministic shuffled batches; a final singleton is merged into the previous
    batch because FC BatchNorm needs at least two images."""
    order = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if len(batches) > 1 and len(batches[-1]) == 1:
        batches[-2].extend(batches.pop())
    return batches


def prepare_eval_batchnorm(model, cfg, loader, image_ids, device):
    """Load or compute BN buffers per ``cfg.bn_calibration_source``.

    ``loader``/``image_ids`` describe the neural dataset's training images and are
    only used by the ``dataset`` source. Sets ``cfg.bn_calibration`` to an identity
    string that becomes part of the results run ID.
    """
    _reject_removed_options(cfg)
    model.eval()
    source = cfg.get("bn_calibration_source", "dataset")
    if source not in SOURCES:
        raise ValueError(f"bn_calibration_source must be one of {SOURCES}, got {source!r}")
    norms = _norm_layers(model)
    if not norms:
        cfg.bn_calibration = "none"
        return model
    if source == "checkpoint":
        cfg.bn_calibration = "checkpoint"
        return model
    if any(not m.track_running_stats for m in norms.values()):
        raise ValueError("BN calibration requires track_running_stats=True")

    if source == "imagenet":
        return _calibrate_on_imagenet(model, cfg, norms, device)
    return _calibrate_on_dataset(model, cfg, norms, loader, image_ids, device)


def _calibrate_on_dataset(model, cfg, norms, loader, image_ids, device):
    image_ids = sorted(set(image_ids))
    if len(image_ids) < 2:
        raise ValueError("BN calibration requires at least two training images")
    batch_size = DATASET_BATCH_SIZE
    metadata = dict(version=1, model=_model_digest(model), dataset=cfg.neural_dataset,
                    image_ids=image_ids, transform=repr(loader.dataset.tr),
                    batch_size=batch_size, shuffle_seed=0, torch_version=str(torch.__version__))

    def make_loader():
        indices = {key: i for i, key in enumerate(loader.dataset.keys)}
        subset = Subset(loader.dataset, [indices[key] for key in image_ids])
        return DataLoader(subset, batch_sampler=_batches(len(subset), batch_size),
                          num_workers=loader.num_workers, collate_fn=loader.collate_fn,
                          pin_memory=loader.pin_memory), len(subset)

    return _load_or_calibrate(model, cfg, norms, metadata, cfg.neural_dataset,
                              make_loader, device)


def _calibrate_on_imagenet(model, cfg, norms, device):
    from visreps.dataloaders.obj_cls import get_obj_cls_loader

    n_images, batch_size = IMAGENET_IMAGES, IMAGENET_BATCH_SIZE
    # The loader picks the backend (and ImageNet release) itself unless overridden.
    data_cfg = {"dataset": "imagenet", "pca_labels": False, "data_augment": False,
                "batchsize": batch_size, "num_workers": int(cfg.get("num_workers", 8))}
    for key in ("imagenet_backend", "imagenet_version"):
        if cfg.get(key) is not None:
            data_cfg[key] = str(cfg.get(key))
    metadata = dict(version=1, model=_model_digest(model), dataset="imagenet",
                    imagenet_version=data_cfg.get("imagenet_version", "default"),
                    n_images=n_images, subset_seed=0, batch_size=batch_size, shuffle_seed=0,
                    torch_version=str(torch.__version__))

    def make_loader():
        datasets, _ = get_obj_cls_loader(data_cfg, shuffle=False)
        train = datasets["train"]
        n = min(n_images, len(train))
        pick = torch.randperm(len(train), generator=torch.Generator().manual_seed(0))[:n]
        subset = Subset(train, sorted(pick.tolist()))
        return DataLoader(subset, batch_sampler=_batches(n, batch_size),
                          num_workers=data_cfg["num_workers"], pin_memory=True), n

    return _load_or_calibrate(model, cfg, norms, metadata, "imagenet", make_loader, device)


def _load_or_calibrate(model, cfg, norms, metadata, tag, make_loader, device):
    """Serve BN buffers from the cache, or compute them (cumulative batch-statistic
    average, no dropout) and cache them. Only BN buffers are stored."""
    identity = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    cfg.bn_calibration = identity
    cache_dir = Path(cfg.get("bn_cache_dir", "model_checkpoints/bn_stats"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{tag}_{identity}.pt"
    if cache.exists():
        saved = torch.load(cache, map_location="cpu", weights_only=True)
        if saved["metadata"] != metadata:
            raise ValueError(f"BN cache metadata mismatch: {cache}")
        for name, module in norms.items():
            for key, value in saved["buffers"][name].items():
                getattr(module, key).copy_(value)
        print(f"  Loaded BN statistics: {cache}")
        return model

    calibration, n_images = make_loader()
    momenta = {name: m.momentum for name, m in norms.items()}
    try:
        for module in norms.values():
            module.reset_running_stats()
            module.momentum = None
            module.train()
        with torch.no_grad():
            for images, _ in calibration:
                model(images.to(device))
    finally:
        for name, module in norms.items():
            module.momentum = momenta[name]
        model.eval()
    buffers = {name: {key: getattr(m, key).detach().cpu().clone()
                      for key in ("running_mean", "running_var", "num_batches_tracked")}
               for name, m in norms.items()}
    if any(not torch.isfinite(value).all() for values in buffers.values() for value in values.values()):
        raise ValueError("Non-finite recalibrated BN statistics")
    fd, temporary = tempfile.mkstemp(dir=cache_dir, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(dict(metadata=metadata, buffers=buffers), temporary)
        os.replace(temporary, cache)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(f"  Calibrated BN on {n_images} {tag} training images: {cache}")
    return model
