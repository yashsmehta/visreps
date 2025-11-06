import os
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pc_i', type=int, required=True)
    parser.add_argument('--csv_file', type=str, required=True)
    args = parser.parse_args()
    
    # Read CSV and filter for PC
    csv_path = os.path.join("datasets/obj_cls/imagenet/pca_poles", args.csv_file)
    df = pd.read_csv(csv_path)
    pc_data = df[df['pc'] == args.pc_i]
    
    # Extract classes for each pole
    low_classes = pc_data[pc_data['pole'] == 'low']['image_class'].tolist()
    high_classes = pc_data[pc_data['pole'] == 'high']['image_class'].tolist()
    
    # Print results
    print(f"PC{args.pc_i}")
    print("LOW (pole):")
    print(','.join(low_classes))
    print()
    print("HIGH (pole):")
    print(','.join(high_classes))

if __name__ == '__main__':
    main() 