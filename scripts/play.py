"""Simple NSD fMRI + stimulus loader."""

import pickle
import h5py
import numpy as np
from PIL import Image
import time
import psutil
import os

from bonner.datasets.allen2021_natural_scenes._stimuli import StimulusSet


def load_fmri(path: str, region: str, subject: int):
    """Load fMRI responses from pickle."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data[region][subject]


def get_pair(fmri_data, stimulus_set: StimulusSet, stimulus_id: int) -> tuple[np.ndarray, Image.Image]:
    """Get (response, image) for a stimulus."""
    response = fmri_data.sel(stimulus=stimulus_id).values
    image = stimulus_set[stimulus_id]
    return response, image

if __name__ == "__main__":
    path = "/data/shared/datasets/allen2021.natural_scenes/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"
    stimulus_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    with h5py.File(path, "r") as f:
        imgBrick = f["imgBrick"]
        
        # Load specific stimulus IDs
        selected_images = imgBrick[stimulus_ids]  # This loads only the images at those indices
        print(f"Selected images shape: {selected_images.shape}")
        print(f"Number of images loaded: {len(selected_images)}")
        
        # You can also use numpy arrays for indexing
        stimulus_array = np.array(stimulus_ids)
        selected_images_np = imgBrick[stimulus_array]
        print(f"Using numpy array indexing: {selected_images_np.shape}")
        
        # Example: load just the first 3 images
        first_three = imgBrick[0:3]  # Slicing works too
        print(f"First three images shape: {first_three.shape}")
        
        # Example: load specific indices (not consecutive)
        specific_ids = [0, 5, 10, 15, 20]
        specific_images = imgBrick[specific_ids]
        print(f"Specific images shape: {specific_images.shape}")
        
        print("\n" + "="*50)
        print("PERFORMANCE TEST: Loading 10,000 random images")
        print("="*50)
        
        # Generate 10,000 random indices from 73,000 total images
        np.random.seed(42)  # For reproducible results
        random_indices = np.random.choice(73000, size=1000, replace=False)
        random_indices = np.sort(random_indices)  # HDF5 requires indices in increasing order
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"Initial memory usage: {initial_memory:.2f} MB")
        print(f"Loading {len(random_indices)} random images...")
        
        # Time the loading
        start_time = time.time()
        random_images = imgBrick[random_indices]
        end_time = time.time()
        
        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = final_memory - initial_memory
        
        print(f"Loading completed!")
        print(f"Time taken: {end_time - start_time:.3f} seconds")
        print(f"Memory used: {memory_used:.2f} MB")
        print(f"Final memory usage: {final_memory:.2f} MB")
        print(f"Images loaded shape: {random_images.shape}")
        print(f"Data type: {random_images.dtype}")
        print(f"Memory per image: {memory_used / len(random_indices):.2f} MB")
        
        # Calculate theoretical memory usage
        theoretical_memory = (425 * 425 * 3 * 10000) / (1024 * 1024)  # bytes to MB
        print(f"Theoretical memory (425x425x3x10000): {theoretical_memory:.2f} MB")

    # fmri_data = load_fmri("/home/ymehta3/research/VisionAI/visreps/datasets/neural/nsd/fmri_responses.pkl", "V1", 0)
    # print(fmri_data.shape)
    # stimulus_set = StimulusSet()
    # print(stimulus_set[0])
    # print(get_pair(fmri_data, stimulus_set, 0))