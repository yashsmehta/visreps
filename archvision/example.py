import pickle
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor

from bonner.computation.metrics import pearson_r
from bonner.computation.regression._linear_regression import LinearRegression

from archvision.utils import setup_logging

# Constants
DATA_DIR = Path('data')
NSD_DATA_FILE = DATA_DIR / 'nsd_data.pkl'
SELECTED_IMAGES_FILE = DATA_DIR / 'selected_images.pkl'
BATCH_SIZE = 32

def load_pickle(file_path):
    """Load data from a pickle file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

class ImageDataset(Dataset):
    """Custom dataset for images stored in a dictionary."""
    def __init__(self, images_dict, transform=None):
        self.image_ids = list(images_dict.keys())
        self.images = images_dict
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image = Image.fromarray(self.images[image_id])
        if self.transform:
            image = self.transform(image)
        return image_id, image

def collate_fn(batch):
    """Custom collate function to handle image IDs and images."""
    image_ids, images = zip(*batch)
    images = torch.stack(images)
    return image_ids, images

def main():
    logger = setup_logging()
    # Load data
    nsd_data = load_pickle(NSD_DATA_FILE)
    selected_images = load_pickle(SELECTED_IMAGES_FILE)

    # Prepare model and transformations
    weights = models.AlexNet_Weights.DEFAULT
    transform = weights.transforms()
    return_nodes = {'classifier.6': 'fc8'}
    model = create_feature_extractor(models.alexnet(weights=weights), return_nodes=return_nodes)

    # Prepare dataset and dataloader
    dataset = ImageDataset(selected_images, transform=transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    logger.info(f"Dataset created with {len(dataset)} images.")

    # Extract activations
    activations_dict = {}
    logger.info("Extracting activations from images...")
    model.eval()
    with torch.no_grad():
        for image_ids, images in dataloader:
            outputs = model(images)['fc8']
            for image_id, activation in zip(image_ids, outputs):
                activations_dict[int(image_id)] = activation.cpu().numpy()
    logger.info("Activations extraction completed.")

    # Process fMRI data
    region_key = 'early visual stream'
    subject_idx = 0

    if region_key not in nsd_data:
        logger.error(f"Region '{region_key}' not found in nsd_data.")
        logger.info(f"Available regions: {list(nsd_data.keys())}")
        return

    subjects_data = nsd_data[region_key]
    stimuli_ids = subjects_data[subject_idx].coords['stimulus'].values
    logger.info(f"Processing data for region '{region_key}', subject {subject_idx} with {len(stimuli_ids)} stimuli.")

    # Prepare X (activations) and Y (fMRI data)
    X = []
    Y = []
    for stimulus_id in stimuli_ids:
        if stimulus_id in activations_dict:
            X.append(activations_dict[stimulus_id])
            Y.append(subjects_data[subject_idx].sel(stimulus=stimulus_id).values)
    logger.info(f"Collected data for {len(X)} valid stimuli.")

    if not X:
        logger.warning(f"No matching stimuli found for subject {subject_idx}.")
        return

    X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
    Y_tensor = torch.tensor(np.array(Y), dtype=torch.float32)
    logger.info(f"Data shapes - X: {X_tensor.shape}, Y: {Y_tensor.shape}")

    # Train Linear Regression model
    lr_model = LinearRegression()
    logger.info("Training linear regression model...")
    lr_model.fit(X_tensor, Y_tensor)

    # Predict and evaluate
    Y_pred = lr_model.predict(X_tensor)
    Y_pred = Y_pred.to(Y_tensor.device)  # Ensure Y_pred is on the same device as Y_tensor
    mse = torch.mean((Y_pred - Y_tensor) ** 2).item()
    r = pearson_r(Y_tensor.cpu(), Y_pred.cpu()).mean().item()  # Move tensors to CPU for pearson_r calculation

    logger.info(f"Model evaluation - MSE: {mse}, Pearson correlation: {r}")
    logger.info("Processing completed.")

if __name__ == "__main__":
    main()
