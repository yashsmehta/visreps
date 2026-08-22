"""Compare visreps and Kazemian et al. NSD encoding-score conventions.

This diagnostic deliberately holds predictions fixed when comparing scores:

* voxelwise: Pearson r over test stimuli for each voxel, then mean voxels;
* pattern: Pearson r over voxels for each test stimulus, then mean stimuli.

The latter reproduces ``scores_tools.pearson_r_`` in
``akazemian/untrained_models_of_visual_cortex``.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.linalg import solve
from torch.utils.data import DataLoader, Dataset
from torchvision.models import AlexNet_Weights, alexnet
from torchvision.transforms import Compose, Normalize, Resize, CenterCrop, ToTensor


ROOT = Path(__file__).resolve().parents[2]
NSD_PKL = ROOT / "datasets/neural/nsd/nsd_data.pkl"
NSD_IMAGES = Path(
    "/data/shared/datasets/allen2021.natural_scenes/"
    "nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"
)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


class NSDImages(Dataset):
    def __init__(self, ids: list[int], preprocessing: str):
        self.ids = ids
        self.h5 = None
        if preprocessing == "visreps":
            self.transform = Compose(
                [Resize(256), CenterCrop(224), ToTensor(), Normalize(MEAN, STD)]
            )
        elif preprocessing == "reference":
            self.transform = Compose(
                [Resize((224, 224)), ToTensor(), Normalize(MEAN, STD)]
            )
        else:
            raise ValueError(preprocessing)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        from PIL import Image

        if self.h5 is None:
            self.h5 = h5py.File(NSD_IMAGES, "r")
        stimulus_id = self.ids[index]
        image = Image.fromarray(self.h5["imgBrick"][stimulus_id]).convert("RGB")
        return self.transform(image), stimulus_id


def extract_fc2(ids, preprocessing, cache, batch_size, workers):
    if cache.exists():
        data = np.load(cache)
        if np.array_equal(data["ids"], ids):
            return data["fc2_pre"], data["fc2_post"]

    model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1).eval()
    captured = {}
    model.classifier[5].register_forward_hook(
        lambda _m, _i, output: captured.__setitem__("pre", output)
    )
    model.classifier[6].register_forward_hook(
        lambda _m, _i, output: captured.__setitem__("post", output)
    )
    loader = DataLoader(
        NSDImages(ids, preprocessing), batch_size=batch_size, shuffle=False,
        num_workers=workers, persistent_workers=workers > 0,
    )
    pre, post = [], []
    with torch.inference_mode():
        for batch_index, (images, _) in enumerate(loader, 1):
            model(images)
            pre.append(captured["pre"].numpy().astype(np.float32))
            post.append(captured["post"].numpy().astype(np.float32))
            if batch_index % 20 == 0:
                print(f"extracted {min(batch_index * batch_size, len(ids))}/{len(ids)}")
    pre, post = np.concatenate(pre), np.concatenate(post)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, ids=np.asarray(ids), fc2_pre=pre, fc2_post=post)
    return pre, post


def corr_axis(y_true, y_pred, axis):
    true = y_true - y_true.mean(axis=axis, keepdims=True)
    pred = y_pred - y_pred.mean(axis=axis, keepdims=True)
    numerator = (true * pred).sum(axis=axis)
    denominator = np.sqrt((true * true).sum(axis=axis) * (pred * pred).sum(axis=axis))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def ridge_predict(x_train, y_train, x_test, alpha):
    """Primal ridge with an unpenalized intercept, matching sklearn Ridge."""
    x_mean = x_train.mean(0, keepdims=True)
    y_mean = y_train.mean(0, keepdims=True)
    xc = x_train - x_mean
    yc = y_train - y_mean
    gram = xc.T @ xc
    gram.flat[:: gram.shape[0] + 1] += alpha
    weights = solve(gram, xc.T @ yc, assume_a="pos", overwrite_a=True)
    return (x_test - x_mean) @ weights + y_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", type=int, default=0)
    parser.add_argument("--preprocessing", choices=["visreps", "reference"], default="visreps")
    parser.add_argument("--fc2", choices=["pre", "post"], default="post")
    parser.add_argument("--alpha", type=float, default=1e5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/visreps_encoding_comparison"))
    args = parser.parse_args()

    with NSD_PKL.open("rb") as handle:
        nsd = pickle.load(handle)
    responses = nsd["data"]["ventral"][args.subject]
    ids = responses.coords["stimulus"].values.astype(int).tolist()
    shared = set(int(i) for i in nsd["shared_ids"])
    train_mask = np.asarray([i not in shared for i in ids])
    test_mask = ~train_mask
    y = responses.values.astype(np.float32)

    cache = args.cache_dir / f"alexnet_fc2_{args.preprocessing}_s{args.subject}.npz"
    pre, post = extract_fc2(ids, args.preprocessing, cache, args.batch_size, args.workers)
    x = pre if args.fc2 == "pre" else post
    prediction = ridge_predict(x[train_mask], y[train_mask], x[test_mask], args.alpha)
    y_test = y[test_mask]

    voxel_scores = corr_axis(y_test, prediction, axis=0)
    pattern_scores = corr_axis(y_test, prediction, axis=1)
    result = {
        "subject": args.subject,
        "roi": "ventral visual stream",
        "layer": f"fc2_{args.fc2}",
        "preprocessing": args.preprocessing,
        "alpha": args.alpha,
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "n_voxels": int(y.shape[1]),
        "voxelwise_mean_r": float(np.nanmean(voxel_scores)),
        "voxelwise_median_r": float(np.nanmedian(voxel_scores)),
        "pattern_mean_r": float(np.nanmean(pattern_scores)),
        "pattern_median_r": float(np.nanmedian(pattern_scores)),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
