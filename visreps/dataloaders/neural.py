import logging
from typing import Tuple, Optional, Dict
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import os

import visreps.utils as utils
from visreps.dataloaders.obj_cls import get_transform

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def select_nsd_subset(nsd_data, cfg):
    """Extracts neural responses for specified region and subject from NSD data"""
    if cfg.region not in nsd_data:
        raise ValueError(f"Region '{cfg.region}' not found. Available: {list(nsd_data.keys())}")
    
    subject_data = nsd_data[cfg.region].get(cfg.subject_idx)
    if subject_data is None:
        raise ValueError(f"Subject {cfg.subject_idx} not found in region {cfg.region}")

    return {str(stim_id): subject_data.sel(stimulus=stim_id).values 
            for stim_id in subject_data.coords["stimulus"].values}

def load_nsd_data(cfg):
    """Loads neural responses and stimuli data"""
    folder = utils.get_env_var("NSD_DATA_DIR")
    fmri_path = os.path.join(folder, "fmri_responses.pkl")
    nsd_data = utils.load_pickle(fmri_path)
    stimuli_path = os.path.join(folder, "stimuli.pkl")
    stimuli = utils.load_pickle(stimuli_path)
    return select_nsd_subset(nsd_data, cfg), stimuli


class StimuliDataset(Dataset):
    """PyTorch dataset for mapping stimulus IDs to image arrays"""

    def __init__(self, stimuli_dict, transform: Optional[transforms.Compose] = None):
        self.transform = transform or transforms.ToTensor()
        self.keys = sorted(stimuli_dict.keys())
        self.images = [stimuli_dict[k] for k in self.keys]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img_array = self.images[idx]
        key = self.keys[idx]
        img_tensor = self.transform(Image.fromarray(img_array.astype("uint8"), "RGB"))
        return img_tensor, key


def custom_collate_fn(batch):
    images, keys = zip(*batch)
    return torch.stack(images, dim=0), list(keys)

def create_nsd_dataloader(
    stimuli_dict,
    transform = None,
    batch_size = 32,
    num_workers = 4,
    shuffle = False
) -> DataLoader:
    dataset = StimuliDataset(stimuli_dict, transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )

def get_neural_loader(cfg: Dict) -> Tuple[Dict, DataLoader]:
    """Prepare neural dataset and its dataloader based on config
    
    Args:
        cfg: Configuration dictionary with:
            - neural_dataset: Which dataset to load ('nsd', etc.)
            - Other dataset-specific parameters
    
    Returns:
        Tuple of (neural_responses, dataloader) where:
            - neural_responses: Dict mapping stimulus IDs to neural responses
            - dataloader: DataLoader for the corresponding stimuli
            
    Raises:
        ValueError: If dataset loading fails or unknown dataset specified
    """
    dataset = cfg.get('neural_dataset', 'nsd')  # Default to NSD for backward compatibility
    
    # Select appropriate data loading function
    if dataset == 'nsd':
        neural_data, stimuli = load_nsd_data(cfg)
        dataloader_fn = create_nsd_dataloader
    # Add more datasets here as needed
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Available: ['nsd']")
    
    if not neural_data or not stimuli:
        raise ValueError(f"Failed to load {dataset} data")
    
    # Create dataloader using dataset-specific function
    dataloader = dataloader_fn(
        stimuli, 
        transform=get_transform(image_size=224),
        batch_size=cfg.batchsize, 
        num_workers=cfg.num_workers
    )
    print(f"Loaded {dataset} data with {len(stimuli)} stimuli")
    return neural_data, dataloader
