import pickle
import xarray as xr
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use the 'Agg' backend which doesn't require GUI
import os

from bonner.datasets.allen2021_natural_scenes import StimulusSet

# Disable custom font installation
import bonner.plotting._fonts
bonner.plotting._fonts.install_newcomputermodern = lambda: None

stimulus_set = StimulusSet()

start_time = time.time()
with open('data/nsd_data.pkl', 'rb') as file:
    data = pickle.load(file)
end_time = time.time()

expected_keys = ["early visual stream", "midventral visual stream", "ventral visual stream"]
expected_subject_indices = list(range(8))  # 0 to 7
data_array = data["early visual stream"][0]
print("Original data_array dimensions:")
print(data_array.dims)

# Create a dictionary to store selected images
selected_images = {}

# Load images corresponding to data_array stimulus values
for stimulus_id in data_array.stimulus.values:
    image = stimulus_set[stimulus_id]
    selected_images[stimulus_id] = np.array(image)

# Save the selected images to a pkl file
with open('data/selected_images.pkl', 'wb') as f:
    pickle.dump(selected_images, f)

print(f"\nTotal number of stimulus IDs: {len(selected_images)}")

load_time = end_time - start_time
print(f"Data loaded and verified successfully in {load_time:.2f} seconds.")
print(f"Selected images have been saved to 'data/selected_images.pkl'.")

print("\nDetailed dimension information of data_array:")
for dim, size in data_array.sizes.items():
    print(f"{dim}: {size}")
