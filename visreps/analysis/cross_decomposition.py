import numpy as np
import pickle
import os
from sklearn.cross_decomposition import PLSSVD
from sklearn.model_selection import KFold
from sklearn.random_projection import GaussianRandomProjection
from visreps.utils import rprint
from tqdm import tqdm


def compute_cross_decomposition_alignment(cfg, activations_dict, neural_data):
    results_file = "logs/eval/cross_decomposition/plssvd_results.pkl"
    os.makedirs("logs/eval/cross_decomposition", exist_ok=True)
    
    # Load existing results if file exists
    if os.path.exists(results_file):
        with open(results_file, 'rb') as f:
            all_results = pickle.load(f)
    else:
        all_results = []

    results = []
    n_folds = 8
    seed = cfg.get('seed')
    checkpoint_epoch = cfg.get('checkpoint_model').split('_')[-1].split('.')[0]
    
    rprint("Computing PLSSVD alignment scores with 8-fold cross-validation...", style="info")
    neural_data = neural_data.cpu().numpy()
    
    neural_projector = GaussianRandomProjection(n_components=1000, random_state=seed)
    activation_projector = GaussianRandomProjection(n_components=1000, random_state=seed)
    
    # Project neural data to 1000d
    neural_data = neural_projector.fit_transform(neural_data)
    
    # Process each layer in return_nodes
    for layer_name, activations in activations_dict.items():
        print(f"\nProcessing Layer: {layer_name}")
        print(f"Activations shape: {activations.shape}")
        
        if activations.ndim > 2:
            activations = activations.flatten(start_dim=1)
        activations = activations.cpu().numpy()
        
        # Project activations to 1000d
        activations = activation_projector.fit_transform(activations)
        
        # Create cross-validation splits
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
        
        # First calculate n_components using first fold to pre-allocate arrays
        first_train_idx, _ = next(kf.split(activations))
        n_components = min(len(first_train_idx), activations.shape[1], neural_data.shape[1])
        
        # Initialize arrays to store results from each fold
        all_correlations = np.zeros((n_folds, n_components))
        all_covariances = np.zeros((n_folds, n_components))
        
        # Perform cross-validation
        for fold_idx, (train_indices, test_indices) in enumerate(tqdm(kf.split(activations), total=n_folds, desc="Processing folds")):
            # Split data
            X_train = activations[train_indices]
            X_test = activations[test_indices]
            y_train = neural_data[train_indices]
            y_test = neural_data[test_indices]
            
            # Calculate n_components for this fold
            fold_n_components = min(X_train.shape[0], X_train.shape[1], y_train.shape[1])
            n_components = min(n_components, fold_n_components)  # Update if needed
            
            # Fit PLSSVD
            pls = PLSSVD(n_components=n_components).fit(X_train, y_train)
            
            # Transform test data
            X_c, y_c = pls.transform(X_test, y_test)
            
            # Compute correlations and covariances
            all_correlations[fold_idx] = np.diag(np.corrcoef(X_c, y_c, rowvar=False)[:n_components, n_components:])
            all_covariances[fold_idx] = np.diag(np.cov(X_c, y_c, rowvar=False)[:n_components, n_components:])
        
        # Compute mean and std across folds
        mean_correlations = np.mean(all_correlations, axis=0)
        mean_covariances = np.mean(all_covariances, axis=0)
        
        result = {
            "layer": layer_name,
            "analysis": "cross_decomposition",
            "mean_correlations": mean_correlations,
            "mean_covariances": mean_covariances,
            "n_components": n_components,
            "n_folds": n_folds,
            "pca_labels": cfg.get('pca_labels'),
            "pca_n_classes": cfg.get('pca_n_classes'),
            "region": cfg.get('region'),
            "epoch": checkpoint_epoch,
            "subject_idx": cfg.get('subject_idx')
        }
        results.append(result)
    
    # Add new results to existing ones
    all_results.extend(results)
    
    # Save updated results
    with open(results_file, 'wb') as f:
        pickle.dump(all_results, f)
    
    rprint("Cross-decomposition alignment scores saved to pickle!", style="success")
    return results