import os
import argparse
import torch
import xarray as xr
from visreps.models import standard_cnn
from visreps.dataloaders.obj_cls import get_obj_cls_loader

def get_activation(name):
    """Returns a hook that saves the output of a module."""
    def hook(module, input, output):
        get_activation.activations[name] = output.detach()
    return hook

def get_model_and_layer_names(dataset):
    """Returns a model and the layer names for feature extraction."""
    model = standard_cnn.AlexNet(
        pretrained_dataset="imagenet1k",
        num_classes=200 if dataset == "tiny-imagenet" else 1000
    )
    return model, ['classifier.5']

def extract_features(model, dataloader, device, layer_names, dataset):
    """Extract features using forward hooks."""
    get_activation.activations = {}
    handles = []
    for name in layer_names:
        module = model
        for part in name.split('.'):
            module = module._modules[part]
        handles.append(module.register_forward_hook(get_activation(name)))

    activations_dict = {name: [] for name in layer_names}
    all_image_names, all_labels = [], []

    for batch_idx, (images, labels, image_names) in enumerate(dataloader):
        print(f"Processing batch {batch_idx+1}/{len(dataloader)}", end='\r')
        images = images.to(device)
        processed_names = [os.path.basename(name) for name in image_names]
        
        with torch.no_grad():
            model(images)

        for name in layer_names:
            activations_dict[name].append(get_activation.activations[name].cpu())
        all_image_names.extend(processed_names)
        all_labels.extend(labels.cpu().numpy())

    for h in handles:
        h.remove()
    for name in layer_names:
        activations_dict[name] = torch.cat(activations_dict[name], dim=0)
    return activations_dict, all_image_names, all_labels

def create_xarray_dataset(activations_dict, image_names, labels, layer_names):
    """Converts activations and metadata into an xarray Dataset."""
    data_vars = {}
    for layer in layer_names:
        acts = activations_dict[layer]
        if acts.ndim > 2:
            acts = acts.reshape(acts.shape[0], -1)
        data_vars[layer] = xr.DataArray(acts.numpy(), dims=['image', 'features'], coords={'image': image_names})
    data_vars['labels'] = xr.DataArray(labels, dims=['image'], coords={'image': image_names})
    return xr.Dataset(data_vars)

class UniformDataLoader:
    """
    Wraps a DataLoader to ensure output in the format (images, labels, image_names).
    Requires original image names from the dataset's samples attribute.
    Raises ValueError if original names are not available.
    """
    def __init__(self, dataloader, dataset_type):
        self.dataloader = dataloader
        self.dataset_type = dataset_type
        self.length = len(dataloader)
        self.current_idx = 0
        
        # Validate that we can get original image names
        if not hasattr(self.dataloader.dataset, 'samples'):
            raise ValueError("Dataset must have 'samples' attribute containing original image names")
        if len(self.dataloader.dataset.samples[0]) <= 2:
            raise ValueError("Dataset samples must contain image paths/names (expected tuple of length > 2)")

    def __len__(self):
        return self.length

    def __iter__(self):
        self.current_idx = 0
        for batch in self.dataloader:
            if len(batch) == 2:
                images, labels = batch
                batch_size = len(images)
                # Get original image names from the dataset's samples
                dataset = self.dataloader.dataset
                image_names = [os.path.basename(dataset.samples[self.current_idx + i][2]) for i in range(batch_size)]
                self.current_idx += batch_size
                yield images, labels, image_names
            else:
                yield batch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, choices=['tiny-imagenet', 'imagenet'],
                        help='Dataset to extract features from')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, layer_names = get_model_and_layer_names(args.dataset)
    model.to(device).eval()

    data_cfg = {
        "dataset": args.dataset,
        "batchsize": 128,
        "num_workers": 8,
        "data_augment": False
    }
    _, loaders = get_obj_cls_loader(data_cfg)
    uniform_loaders = {split: UniformDataLoader(loader, args.dataset) for split, loader in loaders.items()}

    combined_acts = {name: [] for name in layer_names}
    combined_names, combined_labels = [], []

    for split, loader in uniform_loaders.items():
        print(f"\nProcessing {split} split...")
        acts, img_names, labels = extract_features(model, loader, device, layer_names, args.dataset)
        for name in layer_names:
            combined_acts[name].append(acts[name])
        combined_names.extend(img_names)
        combined_labels.extend(labels)

    for name in layer_names:
        combined_acts[name] = torch.cat(combined_acts[name], dim=0)

    print("\nConverting to xarray format...")
    ds = create_xarray_dataset(combined_acts, combined_names, combined_labels, layer_names)
    
    output_dir = os.path.join("datasets", "obj_cls", args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "classification_features.nc")
    print(f"Saving to {output_path}")
    ds.to_netcdf(output_path)

if __name__ == "__main__":
    main()