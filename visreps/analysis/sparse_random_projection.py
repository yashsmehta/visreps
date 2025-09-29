import os
import joblib
import numpy as np
import warnings
from sklearn.random_projection import SparseRandomProjection
from typing import Optional

from visreps.utils import rprint # Assuming rprint is accessible or define a simple print fallback

def _validate_transformer(
    transformer: SparseRandomProjection, 
    k: int, 
    requested_density: Optional[float],
    seed: Optional[int]
) -> bool:
    """
    Checks if a loaded SparseRandomProjection transformer matches the required parameters.

    Args:
        transformer: The loaded transformer instance.
        k: The required number of components (output dimensions).
        requested_density: The required density ('auto' or a float value).
                           If None, density check is skipped (matches 'auto').
        seed: The required random seed.

    Returns:
        True if the transformer is valid, False otherwise.
    """
    if transformer.n_components != k:
        rprint(f"Cached transformer k mismatch (Loaded: {transformer.n_components}, Requested: {k}).", style="warning")
        return False

    loaded_density_value = getattr(transformer, 'density_', None) 
    if loaded_density_value is None:
        rprint(f"Cached transformer seems invalid (no density_ attribute after fit).", style="warning")
        return False
    
    if requested_density is not None and not np.isclose(loaded_density_value, requested_density):
        rprint(f"Cached transformer density mismatch (Loaded: {loaded_density_value:.4f}, Requested: {requested_density:.4f}).", style="warning")
        return False

    loaded_seed = getattr(transformer, 'random_state', None)
    if loaded_seed != seed:
        rprint(f"Cached transformer seed mismatch (Loaded: {loaded_seed}, Requested: {seed}).", style="warning")
        return False
        
    return True

def _fit_new_transformer(
    D: int,
    k: int,
    density: Optional[float],
    seed: Optional[int]
) -> Optional[SparseRandomProjection]:
    """
    Fits a new SparseRandomProjection transformer with the specified parameters.

    Args:
        D: Input dimensionality.
        k: Output dimensionality (number of components).
        density: Density parameter for the projection matrix ('auto' or float).
        seed: Random seed for reproducibility.

    Returns:
        A fitted SparseRandomProjection transformer, or None if fitting fails.
    """
    density_param = density if density is not None else 'auto'
    rprint(f"🔧 Fitting SRP (D={D}→k={k})", style="info")

    try:
        transformer = SparseRandomProjection(
            n_components=k,
            density=density_param,
            random_state=seed
        )
        dummy_data = np.zeros((1, D), dtype=np.float32)
        transformer.fit(dummy_data)
        return transformer
    except Exception as e:
        rprint(f"Failed to fit SRP transformer: {e}", style="error")
        return None

def get_srp_transformer(
    D: int,
    k: int,
    density: Optional[float],
    seed: Optional[int],
    cache_dir: str,
) -> Optional[SparseRandomProjection]:
    """
    Retrieves a cached SparseRandomProjection transformer or fits a new one.

    Ensures the transformer matches the specified dimensions (D, k), density, 
    and random seed. Handles loading from cache, validation, fitting if needed, 
    and caching the new transformer.

    Args:
        D: Input dimensionality.
        k: Output dimensionality (number of components).
        density: Density parameter for the projection matrix ('auto' or float).
        seed: Random seed for reproducibility.
        cache_dir: Directory to store and retrieve cached transformers.

    Returns:
        A validated SparseRandomProjection transformer, or None if invalid parameters
        or fitting fails.
    """
    if k <= 0 or D <= 0:
        rprint(f"Invalid dimensions D={D}, k={k}. Cannot create transformer.", style="error")
        return None
        
    os.makedirs(cache_dir, exist_ok=True)

    density_str = f"{density:.4f}" if density is not None else "auto"
    transformer_filename = f"srp_D{D}_k{k}_density{density_str}_seed{seed}.joblib"
    transformer_path = os.path.join(cache_dir, transformer_filename)

    transformer: Optional[SparseRandomProjection] = None
    should_fit_new = False

    if os.path.exists(transformer_path):
        try:
            loaded_transformer = joblib.load(transformer_path)
            if _validate_transformer(loaded_transformer, k, density, seed):
                transformer = loaded_transformer
                # Silently succeed when loading from cache
            else:
                rprint("Cached transformer validation failed. Will refit.", style="warning")
                should_fit_new = True
        except Exception as e:
            rprint(f"Error loading cached transformer: {e}. Will refit.", style="warning")
            should_fit_new = True
            try:
                os.remove(transformer_path)
            except OSError as remove_err:
                rprint(f"Could not remove problematic cache file: {remove_err}", style="warning")
    else:
        should_fit_new = True 

    if should_fit_new:
        transformer = _fit_new_transformer(D, k, density, seed)

        if transformer is not None:
            try:
                joblib.dump(transformer, transformer_path)
                # Silently cache on success
            except Exception as e:
                rprint(f"Failed to cache transformer: {e}", style="warning")

    return transformer 