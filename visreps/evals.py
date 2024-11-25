import os
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.feature_extraction import create_feature_extractor

import visreps.utils as utils
import visreps.benchmarker as benchmarker
import visreps.metrics as metrics
from visreps.dataloader import get_transform

class StimuliDataset(Dataset):
    def __init__(self, stimuli_dict, transform=None):
        self.transform = transform
        self.keys = sorted(stimuli_dict.keys())
        self.images = [stimuli_dict[k] for k in self.keys]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        key = self.keys[idx]
        img = Image.fromarray(img.astype('uint8'), 'RGB')
        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)
        return img, key

def create_dataloader(stimuli_dict, transform=None, batch_size=32, num_workers=4):
    dataset = StimuliDataset(stimuli_dict, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

def extract_activations(model, dataloader, device):
    model.eval()
    activations_dict = {}
    all_keys = []
    with torch.no_grad():
        for images, keys in dataloader:
            inputs = images.to(device)
            outputs = model(inputs)
            all_keys.extend(keys)
            for node_name, output in outputs.items():
                output = output.cpu()
                activations_dict.setdefault(node_name, []).append(output)
    for node_name in activations_dict:
        activations_dict[node_name] = torch.cat(activations_dict[node_name], dim=0)
    return activations_dict, all_keys

def eval(cfg):
    """
    Evaluate RSA scores for model checkpoints.
    If cfg.checkpoint_path is a file, evaluates only that checkpoint.
    Otherwise, evaluates all checkpoints in the directory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    neural_data, stimuli = benchmarker.load_benchmark_data(cfg)
    transform = get_transform(image_size=64)
    dataloader = create_dataloader(stimuli, transform, batch_size=32, num_workers=4)

    if os.path.isfile(cfg.checkpoint_path):
        checkpoint_files = [cfg.checkpoint_path]
    else:
        checkpoint_dir = cfg.checkpoint_path
        print("Evaluating all checkpoints in:", checkpoint_dir)
        checkpoint_files = sorted(glob.glob(os.path.join(checkpoint_dir, "model_epoch_*.pth")))
        if not checkpoint_files:
            raise ValueError(f"No checkpoint files found in {checkpoint_dir}")
        print(f"Found {len(checkpoint_files)} checkpoint files")

    results_list = []
    for checkpoint_file in checkpoint_files:
        print(f"\nEvaluating checkpoint: {checkpoint_file}")
        
        model = utils.load_model_from_checkpoint(checkpoint_file)
        
        # Prepare the model for feature extraction
        return_nodes = getattr(cfg, 'return_nodes', None)
        if return_nodes:
            model = create_feature_extractor(model, return_nodes=return_nodes)
        else:
            raise ValueError("No return_nodes specified in the configuration for feature extraction.")
        model.to(device)
        
        activations_dict, keys = extract_activations(model, dataloader, device)
 
        # Align activations with neural data
        neural_responses = np.array([neural_data[key] for key in keys])
        
        # Convert activations to NumPy arrays
        for layer in activations_dict:
            activations_dict[layer] = activations_dict[layer].cpu().numpy()
        
        # Better printing
        print("Activations Dictionary:")
        for layer, activations in activations_dict.items():
            print(f"Layer: {layer}, Activations Shape: {activations.shape}")
        
        print("\nNeural Responses:")
        print(f"Shape: {neural_responses.shape}")
        print(neural_responses)
        exit()

        # Calculate RSA scores
        rsa_scores = metrics.calculate_rsa_score(neural_responses, activations_dict)
        
        # Extract epoch number
        try:
            epoch = int(os.path.basename(checkpoint_file).split('_')[-1].split('.')[0])
        except ValueError:
            epoch = None

        results = {
            "epoch": epoch,
            "rsa_scores": rsa_scores,
            "checkpoint_path": checkpoint_file
        }
        
        # Log results
        if getattr(cfg, 'log_expdata', False):
            try:
                utils.log_results(results, folder_name=cfg.exp_name, cfg_id=cfg.cfg_id)
            except (FileNotFoundError, KeyError) as e:
                print(f"An error occurred while logging results: {e}")
        
        results_list.append(results)
        print(f"Epoch {epoch}: RSA Score = {results['rsa_scores']}")
    
    return results_list[0] if len(results_list) == 1 else results_list
