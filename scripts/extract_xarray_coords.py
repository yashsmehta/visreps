import xarray as xr
import pandas as pd
import os

# Define the path to the xarray data
xarray_file_path = '/data/rgautha1/cache/bonner-caching/scale-free-visual-cortex/data/dataset=gifford2025.nsd_synthetic/betas/resolution=1pt8mm/preprocessing=fithrf/z_score=True/roi=general/subject=0.nc'
output_csv_path = 'datasets/neural/nsd_synthetic/extracted_coords.csv'

# Check if the xarray file exists before attempting to load
if not os.path.exists(xarray_file_path):
    print(f"Error: The data file {xarray_file_path} was not found. "
          f"Please ensure the path is correct and the file is accessible.")
    exit()

try:
    # Load the xarray Dataset
    dataset = xr.open_dataset(xarray_file_path)
    
    # Select the 'betas' DataArray (assuming it exists)
    # If 'betas' might not exist or has a different name, error handling for that could be added.
    if 'betas' not in dataset:
        print(f"Error: DataArray 'betas' not found in dataset: {xarray_file_path}")
        exit()
    data_array = dataset['betas']
    
    print(f"Successfully loaded xarray Dataset from {xarray_file_path} and selected 'betas' DataArray")

    # Extract the desired coordinates, assuming they exist at the dataset level
    # and are aligned with the 'presentation' dimension.
    required_coords = ['stimulus', 'subclass', 'class']
    for coord_name in required_coords:
        if coord_name not in dataset.coords:
            print(f"Error: Coordinate '{coord_name}' not found in dataset: {xarray_file_path}")
            exit()

    stimulus_coords = dataset['stimulus'].sel(presentation=data_array.presentation).values
    subclass_coords = dataset['subclass'].sel(presentation=data_array.presentation).values
    class_coords = dataset['class'].sel(presentation=data_array.presentation).values

    # Create a Pandas DataFrame
    df = pd.DataFrame({
        'stimulus': stimulus_coords,
        'subclass': subclass_coords,
        'class': class_coords
    })

    # Ensure uniqueness based on the 'stimulus' column
    df.drop_duplicates(subset=['stimulus'], keep='first', inplace=True)
    print(f"DataFrame now contains {len(df)} unique stimuli after drop_duplicates.")

    # Define desired column order
    column_order = ['class', 'subclass', 'stimulus']
    df = df[column_order]

    # Sort the DataFrame
    df.sort_values(by=['class', 'subclass', 'stimulus'], ascending=[True, True, True], inplace=True)
    print(f"DataFrame sorted by 'class', 'subclass', then 'stimulus'.")

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv_path, index=False)
    print(f"Successfully extracted and processed coordinates to {output_csv_path}")

except FileNotFoundError: # Should be caught by the os.path.exists check, but good for robustness
    print(f"Error: The file {xarray_file_path} was not found during processing.")
except KeyError as e:
    print(f"An error occurred (likely a missing DataArray or Coordinate): {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}") 