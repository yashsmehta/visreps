import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr, pearsonr
import torch

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
    distances = pdist(responses, metric=distance_metric)
    rdm = squareform(distances)
    return rdm


def calculate_rsa_score(
    neural_responses,
    activations,
    distance_metric="euclidean",
    correlation_method="spearman",
):
    neural_rdm = calculate_rdm(neural_responses, distance_metric)
    activation_rdm = calculate_rdm(activations, distance_metric)

    if neural_rdm.shape != activation_rdm.shape:
        raise ValueError(f"RDM shapes do not match: neural RDM shape {neural_rdm.shape}, activation RDM shape {activation_rdm.shape}")

    triu_indices = np.triu_indices_from(neural_rdm, k=1)
    neural_rdm_flat = neural_rdm[triu_indices]
    activation_rdm_flat = activation_rdm[triu_indices]

    if correlation_method == "spearman":
        rsa_score, _ = spearmanr(neural_rdm_flat, activation_rdm_flat)
    elif correlation_method == "pearson":
        rsa_score, _ = pearsonr(neural_rdm_flat, activation_rdm_flat)
    else:
        raise ValueError("Unsupported correlation method. Use 'spearman' or 'pearson'.")

    return rsa_score

def calculate_cls_accuracy(data_loader, model, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total