# Add necessary imports
import visreps.metrics as metrics
import visreps.utils as utils

def load_nsd_data(file_path='benchmarks/nsd/neural_responses.pkl'):
    """
    Load NSD neural responses from a pickle file.

    Args:
        file_path (str): Path to the pickle file containing NSD data.

    Returns:
        dict: Loaded NSD data.
    """
    try:
        return utils.load_pickle(file_path)
    except FileNotFoundError:
        raise FileNotFoundError(f"NSD data file not found at {file_path}")

def prepare_benchmark_data(nsd_data, cfg):
    """
    Prepare benchmark data for a specific region and subject.

    Args:
        nsd_data (dict): NSD data loaded from file.
        cfg (OmegaConf): Configuration object containing experiment details and settings.

    Returns:
        dict: A dictionary containing the benchmark data.
    """
    region_key = cfg.region
    subject_idx = cfg.subject_idx

    if region_key not in nsd_data:
        raise ValueError(f"Region '{region_key}' not found in nsd_data. Available regions: {list(nsd_data.keys())}")

    subjects_data = nsd_data[region_key]
    stimuli_ids = subjects_data[subject_idx].coords['stimulus'].values

    return {stimulus_id: subjects_data[subject_idx].sel(stimulus=stimulus_id).values for stimulus_id in stimuli_ids}

def load_benchmark_data(cfg):
    """
    Load both NSD neural responses and stimuli data.

    Args:
        cfg (OmegaConf): Configuration object containing experiment details and settings.

    Returns:
        tuple: A tuple containing (neural_data, stimuli)
            - neural_data: Dictionary of neural responses
            - stimuli: Dictionary of selected stimuli
    """
    # Load NSD data
    nsd_data = load_nsd_data()
    
    try:
        stimuli = utils.load_pickle('benchmarks/nsd/stimuli.pkl')
    except FileNotFoundError:
        raise FileNotFoundError("Stimuli data file not found")

    # Prepare benchmark data
    neural_data = prepare_benchmark_data(nsd_data, cfg)
    
    return neural_data, stimuli
