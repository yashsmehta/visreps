import os
import torch
import xarray as xr
import argparse
from visreps.models import standard_cnn
from visreps.dataloaders.obj_cls import get_obj_cls_loader

def get_activation(name):
    """Helper function to store layer activations via a forward hook."""
    def hook(model, input, output):
        get_activation.activations[name] = output.detach()
    return hook

def get_model_and_layer_names(dataset):
    """
    Return an appropriate model and the names of the layers whose activations
    will be extracted.
    """
    if dataset == "cifar10":
        model = torch.hub.load(
            "chenyaofo/pytorch-cifar-models",
            "cifar10_resnet20",
            pretrained=True,
            trust_repo=True
        )
        # For CIFAR-10, we use the output of the average pooling layer.
        layer_names = ['avgpool']
    else:  # for tiny-imagenet and imagenet
        model = standard_cnn.AlexNet(
            pretrained_dataset="imagenet1k",
            num_classes=200 if dataset == "tiny-imagenet" else 1000
        )
        # We extract the activation from the ReLU after the fc7 layer.
        layer_names = ['classifier.4']
    return model, layer_names

def extract_features(model, dataloader, device, layer_names, dataset):
    """
    Extract features using forward hooks.
    
    Expects dataloader to return (images, labels, image_names) tuples.
    For CIFAR: image_names are in "class_imagename" format
    For TinyImageNet/ImageNet: image_names are just the base image name
    """
    get_activation.activations = {}
    handles = []
    for name in layer_names:
        # Support nested module lookup (e.g., "classifier.4").
        module = model
        for part in name.split('.'):
            module = module._modules[part]
        handles.append(module.register_forward_hook(get_activation(name)))
    
    activations_dict = {name: [] for name in layer_names}
    all_image_names = []
    all_labels = []
    
    for batch_idx, (images, labels, image_names) in enumerate(dataloader):
        print(f"Processing batch {batch_idx+1}/{len(dataloader)}", end='\r')
        images = images.to(device)
        
        # Process image names based on dataset
        if dataset == "cifar10":
            # Keep original format: class_imagename
            processed_names = image_names
        else:
            # For tiny-imagenet and imagenet: extract just the base image name
            processed_names = [os.path.basename(name) for name in image_names]
            
        with torch.no_grad():
            model(images)
        for name in layer_names:
            acts = get_activation.activations[name]
            activations_dict[name].append(acts.cpu())
        all_image_names.extend(processed_names)
        all_labels.extend(labels.cpu().numpy())
    
    for handle in handles:
        handle.remove()
    
    # Concatenate activations from all batches.
    for name in layer_names:
        activations_dict[name] = torch.cat(activations_dict[name], dim=0)
    
    return activations_dict, all_image_names, all_labels

def create_xarray_dataset(activations_dict, image_names, labels, layer_names):
    """
    Convert activations and associated metadata into an xarray Dataset.
    """
    data_vars = {}
    for layer in layer_names:
        acts = activations_dict[layer]
        if acts.ndim > 2:
            # For convolutional features, flatten spatial dimensions.
            acts = acts.reshape(acts.shape[0], -1)
        data_vars[layer] = xr.DataArray(
            acts.numpy(),
            dims=['image', 'features'],
            coords={'image': image_names}
        )
    data_vars['labels'] = xr.DataArray(
        labels,
        dims=['image'],
        coords={'image': image_names}
    )
    return xr.Dataset(data_vars)

class UniformDataLoader:
    """
    Wrapper for DataLoader that ensures uniform (images, labels, image_names) output.
    Handles both CIFAR (class_idx format) and ImageNet/TinyImageNet (basename format) cases.
    """
    def __init__(self, dataloader, dataset_type):
        self.dataloader = dataloader
        self.dataset_type = dataset_type
        self.cumulative_idx = 0
        self.length = len(dataloader)

    def __len__(self):
        return self.length

    def __iter__(self):
        self.cumulative_idx = 0
        for batch in self.dataloader:
            if len(batch) == 2:  # Only images and labels
                images, labels = batch
                batch_size = len(images)
                
                if self.dataset_type == "cifar10":
                    # For CIFAR: class_imageidx format
                    image_names = [f"{labels[i].item()}_{self.cumulative_idx + i}" for i in range(batch_size)]
                else:
                    # For ImageNet/TinyImageNet: just imageidx
                    image_names = [f"img_{self.cumulative_idx + i}" for i in range(batch_size)]
                
                self.cumulative_idx += batch_size
                yield images, labels, image_names
            else:
                # If already in correct format, process names if needed
                images, labels, image_names = batch
                if self.dataset_type != "cifar10":
                    # For ImageNet/TinyImageNet: strip to basename
                    image_names = [os.path.basename(name) for name in image_names]
                yield images, labels, image_names

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset', type=str, required=True, 
        choices=['cifar10', 'tiny-imagenet', 'imagenet'],
        help='Dataset to extract features from'
    )
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, layer_names = get_model_and_layer_names(args.dataset)
    model.to(device)
    model.eval()
    
    # Data loader configuration
    data_cfg = {
        "dataset": args.dataset,
        "batchsize": 128,
        "num_workers": 8,
        "data_augment": False
    }
    
    _, loaders = get_obj_cls_loader(data_cfg)
    
    # Wrap each loader with UniformDataLoader
    uniform_loaders = {
        split: UniformDataLoader(loader, args.dataset)
        for split, loader in loaders.items()
    }
    
    combined_activations = {name: [] for name in layer_names}
    combined_image_names = []
    combined_labels = []
    
    # Process each split
    for split, loader in uniform_loaders.items():
        print(f"\nProcessing {split} split...")
        activations, image_names, labels = extract_features(model, loader, device, layer_names, args.dataset)
        for name in layer_names:
            combined_activations[name].append(activations[name])
        combined_image_names.extend(image_names)
        combined_labels.extend(labels)
    
    for name in layer_names:
        combined_activations[name] = torch.cat(combined_activations[name], dim=0)
    
    print("\nConverting to xarray format...")
    ds = create_xarray_dataset(combined_activations, combined_image_names, combined_labels, layer_names)
    
    output_dir = os.path.join("datasets", "obj_cls", args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "classification_features.nc")
    print(f"Saving to {output_path}")
    ds.to_netcdf(output_path)

if __name__ == "__main__":
    main()