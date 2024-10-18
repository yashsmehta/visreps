import pickle
import xarray as xr
import time
import numpy as np

start_time = time.time()
with open('nsd_data.pkl', 'rb') as file:
    data = pickle.load(file)
end_time = time.time()

expected_keys = ["early visual stream", "midventral visual stream", "ventral visual stream"]
expected_subject_indices = list(range(8))  # 0 to 7
data_array = data["early visual stream"][0]
print(data_array)

# Print all stimulus IDs
print("All stimulus IDs:")
print(data_array.stimulus.values)

# Print the total number of stimulus IDs
print(f"Total number of stimulus IDs: {len(data_array.stimulus)}")

load_time = end_time - start_time
print(f"Data loaded and verified successfully in {load_time:.2f} seconds.")
