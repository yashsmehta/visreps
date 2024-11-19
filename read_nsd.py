import pickle
import xarray as xr
import time
import numpy as np
import matplotlib
import os

with open('data/nsd_data.pkl', 'rb') as file:
    nsd_data = pickle.load(file)

with open('data/selected_images.pkl', 'rb') as f:
    selected_images = pickle.load(f)

print(selected_images[70758].shape)
print()
print(nsd_data["early visual stream"][0].shape)
