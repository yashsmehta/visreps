#!/usr/bin/env python3
"""Extract representations from supervised ImageNet-pretrained ViT."""

import argparse
import timm
import torch
import torch.nn.functional as F

from utils import get_loaders, extract_features, save_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="imagenet",
        choices=["imagenet", "imagenet-mini-50"],
        help="Dataset to extract features from (default: imagenet)"
    )
    parser.add_argument("--model", type=str, default="vit_large_patch16_224")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load supervised ViT
    model = timm.create_model(args.model, pretrained=True, num_classes=0)
    model.eval().to(device)
    print(f"Loaded {args.model} on {device}")

    # Setup data with model-specific transforms
    loader_list = get_loaders(args.dataset, batch_size=128)
    data_config = timm.data.resolve_model_data_config(model)
    preprocess = timm.data.create_transform(**data_config, is_training=False)
    for loader in loader_list:
        loader.dataset.transform = preprocess

    # Extract CLS token with L2 normalization and save
    def extract_fn(m, x):
        features = m.forward_features(x)[:, 0, :]
        return F.normalize(features, p=2, dim=-1)

    features, image_names = extract_features(
        model, loader_list,
        extract_fn=extract_fn,
        device=device,
        desc=f"Extracting {args.model}"
    )
    save_features(features, image_names, args.dataset, "vit")


if __name__ == '__main__':
    main()
