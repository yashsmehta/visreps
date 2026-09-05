"""Data-driven audit of the hand-assigned ``indoor`` dimension.

The definition says to judge each class by its *typical photo*. This script measures that
directly: for every ImageNet class it samples N images, scores each as indoor vs outdoor
with CLIP zero-shot prompts, and writes the per-class fraction of indoor images next to the
current hand label so disagreements can be reviewed.

Usage (from project root, .env loaded):
    python scripts/coarsegrain/audit_indoor_clip.py --n_per_class 64

Writes ``pca_labels/pca_labels_semantic/audit_indoor_clip.csv``.
"""
import argparse
import os
import random
import warnings

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import clip
import pandas as pd
import torch
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

DIMS_PATH = "pca_labels/pca_labels_semantic/class_dimensions.csv"
OUT_PATH = "pca_labels/pca_labels_semantic/audit_indoor_clip.csv"

INDOOR_PROMPTS = [
    "a photo taken indoors",
    "a photo taken inside a building",
    "a photo taken inside a room of a house",
]
OUTDOOR_PROMPTS = [
    "a photo taken outdoors",
    "a photo taken outside under the sky",
    "a photo taken in nature or on the street",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_class", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    load_dotenv()
    root = os.environ["IMAGENET_DATA_DIR"]
    dims = pd.read_csv(DIMS_PATH)
    device = "cuda"
    model, preprocess = clip.load("ViT-L/14", device=device)
    model.eval()

    with torch.no_grad():
        text = clip.tokenize(INDOOR_PROMPTS + OUTDOOR_PROMPTS).to(device)
        text_feats = model.encode_text(text).float()
        text_feats /= text_feats.norm(dim=-1, keepdim=True)
    n_in = len(INDOOR_PROMPTS)

    rng = random.Random(args.seed)
    indoor_frac, indoor_mean_prob = [], []
    for wnid in tqdm(dims.wnid, desc="CLIP indoor/outdoor"):
        class_dir = os.path.join(root, wnid)
        files = sorted(f for f in os.listdir(class_dir) if f.upper().endswith(".JPEG"))
        files = rng.sample(files, min(args.n_per_class, len(files)))
        batch = torch.stack([preprocess(Image.open(os.path.join(class_dir, f)).convert("RGB")) for f in files])
        with torch.no_grad():
            img_feats = model.encode_image(batch.to(device)).float()
            img_feats /= img_feats.norm(dim=-1, keepdim=True)
            probs = (100.0 * img_feats @ text_feats.T).softmax(dim=-1)
        p_indoor = probs[:, :n_in].sum(dim=1)  # prob mass on the indoor prompts
        indoor_frac.append((p_indoor > 0.5).float().mean().item())
        indoor_mean_prob.append(p_indoor.mean().item())

    out = dims[["class_idx", "wnid", "class_name", "indoor"]].copy()
    out["clip_indoor_frac"] = [round(x, 3) for x in indoor_frac]
    out["clip_indoor_prob"] = [round(x, 3) for x in indoor_mean_prob]
    out["clip_label"] = (out.clip_indoor_frac > 0.5).astype(int)
    out["disagree"] = (out.clip_label != out.indoor).astype(int)
    out.to_csv(OUT_PATH, index=False)
    print(f"saved {OUT_PATH}; {out.disagree.sum()} / {len(out)} classes disagree with the hand label")


if __name__ == "__main__":
    main()
