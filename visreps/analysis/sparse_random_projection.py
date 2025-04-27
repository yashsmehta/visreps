import torch
from typing import Dict, Optional
from sklearn.random_projection import SparseRandomProjection, johnson_lindenstrauss_min_dim
import numpy as np
import warnings

def sparse_random_projection(
    acts: Dict[str, torch.Tensor],
    eps: float = 0.15, # Epsilon for JL lemma
    density: Optional[float] = None,
    seed: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """
    Sparse Johnson-Lindenstrauss projection using scikit-learn.
    Automatically determines target dimensionality k based on JL lemma.

    Args:
        acts: Dictionary {layer_name: activation_tensor (n_samples, n_features)}.
        eps: Maximum distortion allowed by the JL lemma (controls k).
        density: Density of the random matrix. If None, defaults to 'auto' in sklearn,
                 which uses 1/sqrt(n_features).
        seed: Random seed for reproducibility.

    Returns:
        Dictionary {layer_name: projected_activation_tensor (n_samples, k)}.
    """
    if not acts:
        return {}

    projected_acts = {}
    transformer = None
    first_layer_name = next(iter(acts))
    first_act_tensor = acts[first_layer_name]
    n_samples, n_features_original = first_act_tensor.shape[0], first_act_tensor.view(first_act_tensor.shape[0], -1).shape[1]
    original_device = first_act_tensor.device
    original_dtype = first_act_tensor.dtype

    # Calculate minimum dimension k using JL lemma based on n_samples from the first layer
    k = johnson_lindenstrauss_min_dim(n_samples=n_samples, eps=eps)

    # Ensure k is not larger than the smallest feature dimension across all layers
    min_features = min(a.view(a.shape[0], -1).shape[1] for a in acts.values())
    if k > min_features:
        warnings.warn(
            f"JL lemma recommended k={k} > minimum layer features={min_features}. "
            f"Using k={min_features} instead to avoid increasing dimensionality.", RuntimeWarning
        )
        k = min_features

    print(f"JL Lemma (n_samples={n_samples}, eps={eps}): Using target dimension k={k} for all layers.")

    # Initialize and fit the transformer for the first layer's dimension
    current_density = density if density is not None else 'auto'
    transformer_orig = SparseRandomProjection(
        n_components=k,
        density=current_density,
        random_state=seed
    )
    dummy_data_for_fit = np.empty((1, n_features_original), dtype=np.float32)
    transformer_orig.fit(dummy_data_for_fit)

    # Store transformers for different dimensions encountered
    transformers = {n_features_original: transformer_orig}

    for name, a in acts.items():
        if a.ndim < 2:
            raise ValueError(f"{name}: activations must be ≥2-D (got {a.shape}).")

        n, D = a.shape[0], a.view(a.shape[0], -1).shape[1]

        # Get or create transformer for this dimension D
        if D not in transformers:
            # Create and fit a new transformer for this dimension D, projecting to the same k
            current_density_layer = density if density is not None else 'auto'
            transformer_layer = SparseRandomProjection(
                n_components=k, # Use the originally calculated k
                density=current_density_layer,
                random_state=seed
            )
            dummy_data_layer = np.empty((1, D), dtype=np.float32)
            transformer_layer.fit(dummy_data_layer)
            transformers[D] = transformer_layer
            current_transformer = transformer_layer
        else:
            current_transformer = transformers[D]

        # Reshape, move to CPU, convert to NumPy
        v_np = a.view(n, -1).cpu().numpy()

        # Transform
        projected_v_np = current_transformer.transform(v_np)

        # Convert back to PyTorch tensor, move to original device and dtype
        projected_acts[name] = torch.from_numpy(projected_v_np).to(device=original_device, dtype=original_dtype)

    return projected_acts