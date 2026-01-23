#!/usr/bin/env python3
"""Extract FC2 representations from pretrained AlexNet."""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms

from utils import get_loaders, extract_features, save_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="imagenet")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained AlexNet, truncate to FC2
    model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
    model.classifier = nn.Sequential(*list(model.classifier.children())[:6])
    model.eval().to(device)
    print(f"Loaded AlexNet on {device}")

    # Setup data
    loader_list = get_loaders(args.dataset, batch_size=512)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    for loader in loader_list:
        loader.dataset.transform = preprocess

    # Extract with L2 normalization and save
    def extract_fn(m, x):
        features = m(x)
        return F.normalize(features, p=2, dim=-1)

    features, image_names = extract_features(
        model, loader_list,
        extract_fn=extract_fn,
        device=device,
        desc="Extracting AlexNet FC2"
    )
    save_features(features, image_names, args.dataset, "alexnet_features")


if __name__ == '__main__':
    main()
