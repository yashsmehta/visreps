import numpy as np
import json
import os
from sklearn.cross_decomposition import PLSSVD
from sklearn.model_selection import KFold
from visreps.utils import rprint
from plotters.plssvd_plot import plot_binned_correlations
from tqdm import tqdm


def compute_cross_decomposition_alignment(cfg, activations_dict, neural_data):
    results_file = "plotters/plssvd_results.json"
    os.makedirs("plotters", exist_ok=True)
    
    # Load existing results if file exists
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = []

    results = []
    n_folds = 8
    seed = cfg.get('seed')
    
    rprint("Computing PLSSVD alignment scores with 8-fold cross-validation...", style="info")
    
    neural_data = neural_data.cpu().numpy()
    activations = activations_dict['fc2']
    print("\nProcessing Layer: fc2")
    print(f"Activations shape: {activations.shape}")
    
    if activations.ndim > 2:
        activations = activations.flatten(start_dim=1)
    activations = activations.cpu().numpy()
    
    # Initialize arrays to store results from each fold
    n_components = min(activations.shape[1], neural_data.shape[1])
    all_correlations = np.zeros((n_folds, n_components))
    all_covariances = np.zeros((n_folds, n_components))
    
    # Create cross-validation splits
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    # Perform cross-validation
    for fold_idx, (train_indices, test_indices) in enumerate(tqdm(kf.split(activations), total=n_folds, desc="Processing folds")):
        # Split data
        X_train = activations[train_indices]
        X_test = activations[test_indices]
        y_train = neural_data[train_indices]
        y_test = neural_data[test_indices]
        
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
        "layer": 'fc2',
        "analysis": "cross_decomposition",
        "mean_correlations": mean_correlations.tolist(),  # Convert numpy arrays to lists for JSON
        "mean_covariances": mean_covariances.tolist(),
        "n_components": n_components,
        "n_folds": n_folds,
        "pca_labels": cfg.get('pca_labels'),
        "pca_n_classes": cfg.get('pca_n_classes')
    }
    results.append(result)
    
    # Add new results to existing ones
    all_results.extend(results)
    
    # Save updated results
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    exit()
    # save_path = f"plots/plssvd_correlations_fc2_pca{cfg.get('pca_n_classes', 'none')}.png"
    # plot_binned_correlations(mean_correlations, mean_covariances,
    #                        save_path=save_path)
    
    return results