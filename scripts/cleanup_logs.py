import os
import pandas as pd

def cleanup_logs(directory):
    """
    Cleans up log files in the specified directory by filtering out specific layers.

    Parameters:
    directory (str): The path to the directory containing the log files. Defaults to 'logs/arch_params'.

    The function performs the following steps for each CSV file in the directory:
    1. Reads the CSV file into a pandas DataFrame.
    2. Filters out rows where the 'model_layer' column contains 'Sequential', 'BaseCNN', 'Dropout', or 'BatchNorm'.
    3. Converts the 'model_layer_index' column to float.
    4. Normalizes the 'model_layer_index' column by dividing by its maximum value.
    5. Saves the modified DataFrame back to the CSV file.
    """
    for idx, filename in enumerate(os.listdir(directory)):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            print(f"File {idx + 1}: {filename}")
            df = pd.read_csv(file_path)
            df = df[~df['model_layer'].str.contains('Sequential|BaseCNN|Dropout|BatchNorm')]
            df['model_layer_index'] = df['model_layer_index'].astype(float)
            df['model_layer_index'] = df['model_layer_index'] / df['model_layer_index'].max()
            df.to_csv(file_path, index=False)


if __name__ == "__main__":
    exp_name = input("Enter the name of the experiment: ")
    cleanup_logs(f"logs/{exp_name}")

