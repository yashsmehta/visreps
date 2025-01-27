import xarray as xr
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import os
import pandas as pd

# Number of bits for classification (will create 2^n classes)
n_bits = 3  # This will create 2^n_bits classes. Change this to get different numbers of classes

# Load the features
features_path = "datasets/obj_cls/cifar10/classification_features.nc"
base_dir = os.path.dirname(features_path)
labels_dir = os.path.join(base_dir, "pca_labels")
os.makedirs(labels_dir, exist_ok=True)

ds = xr.open_dataset(features_path)

# Print dataset info
print("Dataset structure:")
print(ds)

# Get image names
image_names = ds.image.values

# Convert to numpy array and reshape for PCA
feature_array = ds.avgpool.values
n_samples = feature_array.shape[0]
features_2d = feature_array.reshape(n_samples, -1)

# Print shape info
print(f"\nFeature array shape: {feature_array.shape}")
print(f"Reshaped features shape: {features_2d.shape}")

# Standardize the features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_2d)

# Perform PCA
pca = PCA()
pc_scores = pca.fit_transform(features_scaled)

# Get first n PC scores
pc_scores_n = pc_scores[:, :n_bits]

# Create binary classifications for each PC
binary_labels = []
for i in range(n_bits):
    pc_i_scores = pc_scores[:, i]
    median_pc = np.median(pc_i_scores)
    binary_labels.append((pc_i_scores > median_pc).astype(int))

# Convert binary labels to decimal class labels (0 to 2^n - 1)
binary_labels = np.array(binary_labels).T  # Shape: (n_samples, n_bits)
class_labels = np.zeros(n_samples, dtype=int)
for i in range(n_bits):
    class_labels += binary_labels[:, i] * (2 ** (n_bits - 1 - i))

# Print statistics
print(f"\nUsing first {n_bits} PCs to create {2**n_bits} classes")
for i in range(n_bits):
    print(f"PC{i+1} explains {pca.explained_variance_ratio_[i]*100:.2f}% of variance")
print("\nClass distribution:")
for i in range(2**n_bits):
    print(f"Class {i}: {np.sum(class_labels == i)} samples")

# Save results as DataFrame
df = pd.DataFrame({
    'image': image_names,
    'label': class_labels
})

# Save DataFrame as CSV
output_csv_file = os.path.join(labels_dir, f"n_bits_{n_bits}.csv")
df.to_csv(output_csv_file, index=False)

