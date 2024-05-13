import os
import pandas as pd

def consolidate_csv_files(folder_path, output_file):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The folder path {folder_path} does not exist.")
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    dataframes = []
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path, dtype={'conv_trainable': 'string', 'fc_trainable': 'string'})
        dataframes.append(df)
    consolidated_df = pd.concat(dataframes, ignore_index=True)
    consolidated_df = consolidated_df.drop_duplicates()
    consolidated_df.to_csv(output_file, index=False)
    print(f"Number of files consolidated: {len(csv_files)}")
    print(f"Consolidated CSV written to {output_file}")

if __name__ == "__main__":
    exp_name = input("Please enter the experiment name: ")
    consolidate_csv_files(f'logs/{exp_name}', f'logs/{exp_name}.csv')

