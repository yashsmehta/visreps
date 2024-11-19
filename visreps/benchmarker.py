# Add necessary imports
import visreps.metrics as metrics
import visreps.utils as utils

def load_benchmark(cfg):
    """
    Load and prepare the benchmark data using the configuration.

    Args:
        cfg (OmegaConf): Configuration object containing experiment details and settings.

    Returns:
        dict: A dictionary containing the benchmark data.
    """
    # Load NSD neural responses
    nsd_data = utils.load_pickle('data/nsd/neural_responses.pkl')

    # Process fMRI data and prepare for benchmarking
    region_key = cfg.region
    subject_idx = cfg.subject_idx

    if region_key not in nsd_data:
        raise ValueError(f"Region '{region_key}' not found in nsd_data. Available regions: {list(nsd_data.keys())}")

    subjects_data = nsd_data[region_key]
    stimuli_ids = subjects_data[subject_idx].coords['stimulus'].values

    benchmark = {}
    for stimulus_id in stimuli_ids:
        benchmark[stimulus_id] = subjects_data[subject_idx].sel(stimulus=stimulus_id).values

    return benchmark

def get_benchmarking_results(benchmark, activations_dict):
    """
    Calculate benchmarking results using the RSA score.

    Args:
        benchmark (dict): The benchmark data.
        activations_dict (dict): The model activations.

    Returns:
        dict: A dictionary containing the RSA scores.
    """
    # Calculate RSA score
    rsa_scores = metrics.calculate_rsa_score(benchmark, activations_dict)

    # Prepare results
    results = {
        "rsa_scores": rsa_scores,
        "metric": "rsa"
    }

    return results 