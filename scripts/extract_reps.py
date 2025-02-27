import os
import argparse
import torch
import numpy as np
from omegaconf import OmegaConf
from visreps.models import standard_cnn
from visreps.models.utils import configure_feature_extractor
from visreps.dataloaders.obj_cls import get_obj_cls_loader

def get_model_and_layers(dataset, pretrained_dataset="imagenet1k"):
    """Initializes the model and defines the layers to extract features from.
    """
    num_classes = 200 if dataset == "tiny-imagenet" else 1000
    model = standard_cnn.AlexNet(pretrained_dataset=pretrained_dataset, num_classes=num_classes)
    
    layers = [
        'fc2'
    ]
    return model, layers

def extract_features(model, loader, device, layer_names):
    """Extracts activations using the feature extractor."""
    acts_dict = {name: [] for name in layer_names}
    all_image_names, all_labels = [], []

    for batch_idx, (images, labels) in enumerate(loader):
        print(f"Processing batch {batch_idx+1}/{len(loader)}", end='\r')
        images = images.to(device)

        # Get image names from dataset
        start = batch_idx * loader.batch_size
        end = start + images.size(0)
        names = [os.path.basename(loader.dataset.samples[i][2]) for i in range(start, end)]
        
        with torch.no_grad():
            features = model(images)  # Returns dict of layer activations

        # Save each layer's activation
        for name in layer_names:
            acts_dict[name].append(features[name].cpu().numpy())
        all_image_names.extend(names)
        all_labels.extend(labels.cpu().numpy())

    # Concatenate activations from all batches
    for name in layer_names:
        acts_dict[name] = np.concatenate(acts_dict[name], axis=0)
    return acts_dict, all_image_names, all_labels

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['tiny-imagenet', 'imagenet'],
                        help='Dataset to extract features from')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_dataset = "none"
    model, layers = get_model_and_layers(args.dataset, pretrained_dataset)
    
    # Configure feature extractor with OmegaConf
    cfg = OmegaConf.create({"return_nodes": layers})
    feature_extractor = configure_feature_extractor(cfg, model)
    feature_extractor.to(device).eval()

    data_cfg = {
        "dataset": args.dataset,
        "batchsize": 512, 
        "num_workers": 8,
        "data_augment": False
    }
    # Important to set shuffle to False, so that the order of the features is the same as the order of the images!
    _, loaders = get_obj_cls_loader(data_cfg, shuffle=False)

    combined_features = {}
    combined_names = []
    combined_labels = []

    # Extract features from all splits
    for split, loader in loaders.items():
        print(f"\nProcessing {split} split...")
        acts, names, labels = extract_features(feature_extractor, loader, device, layers)
        
        # First time initialization
        if not combined_features:
            combined_features = {layer: [] for layer in acts.keys()}
        
        # Append features from this split
        for layer in acts:
            combined_features[layer].append(acts[layer])
        combined_names.extend(names)
        combined_labels.extend(labels)

    # Concatenate all splits
    for layer in combined_features:
        combined_features[layer] = np.concatenate(combined_features[layer], axis=0)
        print(f"{layer}: {combined_features[layer].shape}")

    output_dir = os.path.join("datasets", "obj_cls", args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"features_pretrained_{pretrained_dataset}.npz")
    
    # Save combined features
    save_dict = {
        'image_names': combined_names,
        'labels': combined_labels
    }
    save_dict.update(combined_features)
    
    np.savez(output_path, **save_dict)
    print(f"Features saved to {output_path}")

if __name__ == '__main__':
    main()