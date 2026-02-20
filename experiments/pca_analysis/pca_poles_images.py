import os
import argparse
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from visreps import utils

def load_imagenet_class_mapping(imagenet_data_dir):
    """Load ImageNet class ID to class name mapping from map_clsloc.txt."""
    mapping_file = os.path.join(imagenet_data_dir, "map_clsloc.txt")
    class_mapping = {}
    with open(mapping_file, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split(' ', 1)
                if len(parts) >= 2:
                    class_mapping[parts[0]] = parts[1]
    print(f"Loaded {len(class_mapping)} class mappings")
    return class_mapping

def analyze_pc_poles(pc_scores, image_names, class_mapping, n_poles=100):
    """Analyze images at opposite poles of each principal component."""
    all_poles_data = []
    
    for pc_idx in range(pc_scores.shape[1]):
        pc_scores_1d = pc_scores[:, pc_idx]
        sorted_indices = np.argsort(pc_scores_1d)
        
        poles = [
            (sorted_indices[:n_poles], 'low'),
            (sorted_indices[-n_poles:][::-1], 'high')
        ]
        
        for indices, pole_name in poles:
            for idx in indices:
                image_name = image_names[idx]
                class_id = image_name.split('_')[0]
                all_poles_data.append({
                    'pc': pc_idx + 1,
                    'pole': pole_name,
                    'score': pc_scores_1d[idx],
                    'image_file': image_name,
                    'image_class_id': class_id,
                    'image_class': class_mapping.get(class_id, "unknown")
                })
    
    return pd.DataFrame(all_poles_data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_filename', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='imagenet')
    args = parser.parse_args()

    # Load class mapping and data
    class_mapping = load_imagenet_class_mapping(utils.get_env_var("IMAGENET_DATA_DIR"))
    features_path = os.path.join("datasets", "obj_cls", args.dataset, args.features_filename)
    data = np.load(features_path, allow_pickle=True)
    print(f"Loaded features from {features_path}")
    
    # Get features
    features = data['fc2'] if 'fc2' in data else data['clip_features']
    image_names = [os.path.basename(str(name)) for name in data['image_names']]
    
    if features.ndim != 2:
        features = features.reshape(features.shape[0], -1)
    
    # Run PCA
    print("Running PCA...")
    features_scaled = StandardScaler().fit_transform(features)
    n_fit = min(110000, features.shape[0])
    np.random.seed(42)
    fit_indices = np.random.choice(features.shape[0], n_fit, replace=False)
    
    pca = PCA(n_components=6)
    pca.fit(features_scaled[fit_indices])
    pc_scores = pca.transform(features_scaled)
    print("PCA completed")
    
    # Analyze poles and save results
    df = analyze_pc_poles(pc_scores, image_names, class_mapping)
    
    features_suffix = args.features_filename.replace('features_', '').replace('.npz', '')
    output_dir = os.path.join("datasets", "obj_cls", args.dataset, f"pca_poles")
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, f"pca_poles_{features_suffix}.csv")
    df.to_csv(csv_file, index=False)
    print(f"Saved results to {csv_file}")

if __name__ == '__main__':
    main()