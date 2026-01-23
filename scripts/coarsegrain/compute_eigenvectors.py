"""
Compute eigenvectors (principal components) from extracted features.

This script performs PCA on fc2 features and saves the top N eigenvectors
for use in generating coarse-grained PCA labels.

Note on Feature Source:
    For AlexNet, the features file `features_alexnet.npz` was extracted using
    PRETRAINED torchvision AlexNet (ImageNet1K weights), containing fc2
    activations for 1.26M ImageNet training images.
"""
import os
import numpy as np

# Configuration
model_name='vit'
# Note: For alexnet, features extracted from PRETRAINED torchvision AlexNet
FEATURES_PATH = f"datasets/obj_cls/imagenet/features_{model_name}.npz"
OUTPUT_PATH = f"datasets/obj_cls/imagenet/eigenvectors_{model_name}.npz"
N_COMPONENTS = 20 # num of PCs to save (resulting num classes = 2^N_COMPONENTS)
BATCH_SIZE = 10000

def batched_pca(X, n_components, batch_size):
    """Exact PCA via batched covariance computation."""
    n, p = X.shape
    mean = X.mean(axis=0)
    
    cov = np.zeros((p, p), dtype=np.float64)
    print(f"Iterating through {n} samples in batches of {batch_size}...")
    for i in range(0, n, batch_size):
        batch = X[i:i + batch_size].astype(np.float64) - mean
        cov += batch.T @ batch
        if (i // batch_size) % 10 == 0:
            print(f"  Processed {i}/{n}...")
            
    cov /= (n - 1)
    
    print("Computing eigendecomposition...")
    vals, vecs = np.linalg.eigh(cov)
    # Sort descending
    idx = np.argsort(vals)[::-1][:n_components]
    total_var = vals.sum()
    
    return vecs[:, idx], vals[idx], mean, total_var

def main():
    print(f"Loading features from {FEATURES_PATH}...")
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    
    data = np.load(FEATURES_PATH, allow_pickle=True)
    features = data[f'{model_name}_features']

    print(f"Features shape: {features.shape}")
    components, eigenvalues, mean, total_var = batched_pca(features, N_COMPONENTS, BATCH_SIZE)
    
    # Save the model
    np.savez(OUTPUT_PATH, 
             eigenvectors=components, 
             eigenvalues=eigenvalues, 
             mean=mean,
             total_variance=total_var)
    
    print(f"Eigenvectors saved to {OUTPUT_PATH}")
    var_exp = (eigenvalues[:6].sum() / total_var) * 100
    print(f"Variance explained by top 6: {var_exp:.2f}%")

if __name__ == '__main__':
    main()