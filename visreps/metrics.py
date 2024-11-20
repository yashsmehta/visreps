import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr, pearsonr

def calculate_rdm(data, distance_metric='euclidean'):
    """
    Calculate the representational dissimilarity matrix (RDM) for given data.

    Args:
        data (np.ndarray): Data for which to calculate the RDM.
        distance_metric (str): Metric used to compute the RDM (default 'euclidean').

    Returns:
        np.ndarray: The RDM of the data.
    """
    return pdist(data, metric=distance_metric)

def calculate_rsa_score(benchmark, activations_dict, distance_metric='euclidean', correlation_method='spearman'):
    """
    Calculate the RSA score between model activations and neural responses.

    Args:
        benchmark (dict): A dictionary where keys are stimulus IDs and values are neural responses.
        activations_dict (dict): A dictionary where keys are stimulus IDs and values are model activations.
        distance_metric (str): Metric used to compute the RDM (default 'euclidean').
        correlation_method (str): Method to compute correlation ('spearman' or 'pearson').

    Returns:
        float: The RSA correlation coefficient.
    """
    common_stimuli = set(benchmark.keys()).intersection(activations_dict.keys())
    neural_responses = np.array([benchmark[stimulus_id] for stimulus_id in common_stimuli])
    neural_activations = np.array([activations_dict[stimulus_id] for stimulus_id in common_stimuli])

    neural_rdm = calculate_rdm(neural_responses, distance_metric)
    activation_rdm = calculate_rdm(neural_activations, distance_metric)

    if correlation_method == 'spearman':
        rsa_score, _ = spearmanr(neural_rdm, activation_rdm)
    elif correlation_method == 'pearson':
        rsa_score, _ = pearsonr(neural_rdm, activation_rdm)
    else:
        raise ValueError("Unsupported correlation method. Use 'spearman' or 'pearson'.")

    return rsa_score
