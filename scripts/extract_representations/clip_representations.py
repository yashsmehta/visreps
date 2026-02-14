#!/usr/bin/env python3
"""Extract representations from CLIP."""

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import argparse
import clip
import torch

from utils import get_loaders, extract_features, save_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="imagenet",
        choices=["imagenet", "imagenet-mini-50"],
        help="Dataset to extract features from (default: imagenet)"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load CLIP
    model, preprocess = clip.load("ViT-L/14", device=device)
    model.eval()
    print(f"Loaded CLIP ViT-L/14 on {device}")

    # Setup data
    loader_list = get_loaders(args.dataset, batch_size=128)
    for loader in loader_list:
        loader.dataset.transform = preprocess

    # Extract with L2 normalization
    def extract_fn(m, x):
        features = m.encode_image(x)
        return features / features.norm(dim=-1, keepdim=True)

    features, image_names = extract_features(
        model, loader_list,
        extract_fn=extract_fn,
        device=device,
        desc="Extracting CLIP"
    )
    save_features(features, image_names, args.dataset, "clip")


if __name__ == '__main__':
    main()
