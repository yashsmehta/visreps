"""Training-image-only BatchNorm calibration for evaluation."""
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset


def training_image_ids(neural, available):
    """Union of training images, excluding every participating subject's test set."""
    train, test = set(), set()
    for subjects in neural.values():
        for splits in subjects.values():
            train.update(splits["train"])
            test.update(splits["test"])
    return sorted((train - test) & set(available))


def prepare_eval_batchnorm(model, cfg, loader, image_ids, device):
    """Load or compute BN buffers; leave weights fixed and the model in eval mode.

    Calibration uses deterministic shuffled batches, cumulative batch-statistic
    averages, and no dropout. A final singleton is merged into the previous batch
    because FC BatchNorm needs at least two images. Only BN buffers are cached.
    """
    model.eval()
    norms = {name: m for name, m in model.named_modules()
             if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))}
    if not norms:
        cfg.bn_calibration = "none"
        return model
    if any(not m.track_running_stats for m in norms.values()):
        raise ValueError("BN calibration requires track_running_stats=True")
    image_ids = sorted(set(image_ids))
    if len(image_ids) < 2:
        raise ValueError("BN calibration requires at least two training images")
    batch_size = int(cfg.get("bn_calibration_batchsize", 64))
    if batch_size < 2:
        raise ValueError("bn_calibration_batchsize must be at least 2")

    # Hash the original model, including buffers, so seeds/epochs cannot collide.
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        value = value.detach().cpu().contiguous()
        digest.update(str((name, str(value.dtype), tuple(value.shape))).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    metadata = dict(version=1, model=digest.hexdigest(), dataset=cfg.neural_dataset,
                    image_ids=image_ids, transform=repr(loader.dataset.tr),
                    batch_size=batch_size, shuffle_seed=0, torch_version=str(torch.__version__))
    identity = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()
    cfg.bn_calibration = identity
    cache_dir = Path(cfg.get("bn_cache_dir", "model_checkpoints/bn_stats"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{cfg.neural_dataset}_{identity}.pt"
    if cache.exists():
        saved = torch.load(cache, map_location="cpu", weights_only=True)
        if saved["metadata"] != metadata:
            raise ValueError(f"BN cache metadata mismatch: {cache}")
        for name, module in norms.items():
            for key, value in saved["buffers"][name].items():
                getattr(module, key).copy_(value)
        print(f"  Loaded BN statistics: {cache}")
        return model

    indices = {key: i for i, key in enumerate(loader.dataset.keys)}
    subset = Subset(loader.dataset, [indices[key] for key in image_ids])
    order = torch.randperm(len(subset), generator=torch.Generator().manual_seed(0)).tolist()
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if len(batches[-1]) == 1:
        batches[-2].extend(batches.pop())
    calibration = DataLoader(subset, batch_sampler=batches, num_workers=loader.num_workers,
                             collate_fn=loader.collate_fn, pin_memory=loader.pin_memory)
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
    print(f"  Calibrated BN on {len(image_ids)} training images: {cache}")
    return model
