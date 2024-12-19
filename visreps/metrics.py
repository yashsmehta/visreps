import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, pearsonr

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr


def calculate_rdm(responses, distance_metric="euclidean"):
    """
    Calculate the Representational Dissimilarity Matrix (RDM).

    Args:
        responses (np.ndarray): Responses (e.g., neural responses or activations).
                                Shape: (N stimuli, D features)
        distance_metric (str): Metric used to compute the RDM.

    Returns:
        np.ndarray: The RDM matrix of shape (N, N).
    """
    # Compute the pairwise distances
    distances = pdist(responses, metric=distance_metric)
    # Convert to a square matrix form
    rdm = squareform(distances)
    return rdm


def calculate_rsa_score(
    neural_responses,
    activations,
    distance_metric="euclidean",
    correlation_method="spearman",
):
    """
    Calculate the RSA score between model activations and neural responses.

    Args:
        neural_responses (np.ndarray): Neural responses. Shape: (N stimuli, D features)
        activations (np.ndarray): Model activations. Shape: (N stimuli, D features)
        distance_metric (str): Metric used to compute the RDM (default 'euclidean').
        correlation_method (str): Method to compute correlation ('spearman' or 'pearson').

    Returns:
        float: The RSA correlation coefficient.
    """
    # Calculate RDMs
    neural_rdm = calculate_rdm(neural_responses, distance_metric)
    print(f"Neural RDM Shape: {neural_rdm.shape}")
    activation_rdm = calculate_rdm(activations, distance_metric)
    print(f"Activation RDM Shape: {activation_rdm.shape}")

    # Flatten the upper triangle of the RDMs, excluding the diagonal
    triu_indices = np.triu_indices_from(neural_rdm, k=1)
    neural_rdm_flat = neural_rdm[triu_indices]
    activation_rdm_flat = activation_rdm[triu_indices]

    # Compute the correlation between the flattened RDMs
    if correlation_method == "spearman":
        rsa_score, _ = spearmanr(neural_rdm_flat, activation_rdm_flat)
    elif correlation_method == "pearson":
        rsa_score, _ = pearsonr(neural_rdm_flat, activation_rdm_flat)
    else:
        raise ValueError("Unsupported correlation method. Use 'spearman' or 'pearson'.")

    return rsa_score
