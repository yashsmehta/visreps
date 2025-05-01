import torch, numpy as np
from sklearn.decomposition import PCA
from typing import Dict, Union

Array = Union[torch.Tensor, np.ndarray]

def reconstruct_from_pcs(acts: Dict[str, Array], k: int) -> Dict[str, Array]:
    """Return activations reconstructed from their top-k PCs, preserving input types."""
    out = {}
    for name, x in acts.items():
        # --- sanity checks & to-numpy -------------------------------------------------
        if isinstance(x, torch.Tensor):
            if x.ndim < 2: raise ValueError(f"{name}: need ≥2-D tensor")
            dev, dt = x.device, x.dtype
            x_np = x.detach().cpu().numpy()
            to_orig = lambda a: torch.as_tensor(a, dtype=dt, device=dev)
        elif isinstance(x, np.ndarray):
            if x.ndim < 2: raise ValueError(f"{name}: need ≥2-D array")
            x_np = x
            to_orig = lambda a: a
        else:
            raise TypeError(f"{name}: expect torch.Tensor or np.ndarray")

        # --- PCA ----------------------------------------------------------------------
        flat = x_np.reshape(x_np.shape[0], -1)
        pca  = PCA(n_components=min(k, flat.shape[1]))
        rec  = pca.inverse_transform(pca.fit_transform(flat))

        out[name] = to_orig(rec)
    return out