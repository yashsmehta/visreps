"""
Binary PC RSA Analysis

Computes RSA between binary PC codes (from AlexNet fc2) and NSD neural data.

Note: Eigenvectors computed from PRETRAINED torchvision AlexNet (ImageNet1K weights).
"""

import os
import sys

import numpy as np
import pandas as pd
import scipy.stats
import torch
from torchvision.models import alexnet, AlexNet_Weights
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visreps.dataloaders.neural import load_nsd_data, _make_loader
from visreps.dataloaders.obj_cls import get_transform
from visreps.models.utils import FeatureExtractor
from visreps.analysis.rsa import compute_rsm, _kendall_tau_a


def _compare_rsms(rsm1: torch.Tensor, rsm2: torch.Tensor, correlation: str) -> float:
    """Compare upper triangular parts of two RSMs."""
    n = rsm1.size(0)
    idx = torch.triu_indices(n, n, offset=1)
    v1, v2 = rsm1[idx[0], idx[1]].cpu().numpy(), rsm2[idx[0], idx[1]].cpu().numpy()

    if correlation.lower() == "kendall":
        return _kendall_tau_a(v1, v2)[0]
    elif correlation.lower() == "spearman":
        return scipy.stats.spearmanr(v1, v2).statistic
    return scipy.stats.pearsonr(v1, v2).statistic

# ============ CONFIGURATION (edit these) ============
N_PCS_LIST = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
SUBJECT_IDX_LIST = [0,1]
WEIGHTED = [True, False]  # True, False, or [True, False]
RSM_CORRELATION = ["Spearman", "Kendall"]  # "Spearman", "Kendall", "Pearson", or list
OUTPUT_PATH = "experiments/results/binary_pc_rsa.csv"
# ====================================================

EIGENVECTORS_PATH = "datasets/obj_cls/imagenet/eigenvectors_alexnet.npz"
REGIONS = ["early visual stream", "ventral visual stream"]


def compute_hamming_rsm(binary_codes: np.ndarray, weighted: bool = True) -> torch.Tensor:
    """Compute Hamming similarity RSM for binary codes.
    
    Args:
        binary_codes: (n_images, n_bits) binary array
        weighted: If True, weight by inverse rank (PC1 > PC2 > ... > PCn)
    """
    codes = torch.from_numpy(binary_codes).float()
    n_images, n_bits = codes.shape
    xor = (codes.unsqueeze(1) != codes.unsqueeze(0)).float()  # (n_images, n_images, n_bits)
    
    if weighted:
        # Inverse rank: PC1 gets weight n_bits, PC_n gets weight 1
        weights = torch.arange(n_bits, 0, -1, dtype=torch.float32)
        weighted_dist = (xor * weights).sum(dim=2)
        return 1 - weighted_dist / weights.sum()
    else:
        hamming_dist = xor.sum(dim=2)
        return 1 - hamming_dist / n_bits


def project_and_binarize(activations: np.ndarray, eigenvectors: np.ndarray,
                         mean: np.ndarray, n_pcs: int) -> np.ndarray:
    """Project activations onto PCs and apply median split."""
    centered = activations - mean
    pc_scores = centered @ eigenvectors[:, :n_pcs]
    medians = np.median(pc_scores, axis=0)
    return (pc_scores > medians).astype(np.int32)


def extract_fc2_activations(model: torch.nn.Module, dataloader, device) -> tuple:
    """Extract fc2 activations from AlexNet."""
    extractor = FeatureExtractor(model, return_nodes={"fc2": "fc2"})
    extractor.eval()

    all_activations, all_ids = [], []
    with torch.no_grad():
        for imgs, keys in tqdm(dataloader, desc="Extracting activations"):
            features = extractor(imgs.to(device))
            fc2_out = features["fc2"]
            if fc2_out.ndim > 2:
                fc2_out = fc2_out.flatten(start_dim=1)
            all_activations.append(fc2_out.cpu().numpy())
            all_ids.extend(keys)

    return np.concatenate(all_activations, axis=0), all_ids


def align_data(activations: np.ndarray, activation_ids: list,
               neural_data: dict) -> tuple:
    """Align activations with neural data by stimulus ID."""
    common_ids = sorted(set(activation_ids) & set(neural_data.keys()))
    id_to_idx = {id_: idx for idx, id_ in enumerate(activation_ids)}
    aligned_acts = np.array([activations[id_to_idx[id_]] for id_ in common_ids])
    aligned_neural = np.array([neural_data[id_] for id_ in common_ids])
    return aligned_acts, aligned_neural


def run_analysis():
    """Run binary PC RSA analysis for all configurations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Binary PC RSA | n_pcs={N_PCS_LIST}, subjects={SUBJECT_IDX_LIST}, device={device}\n")

    model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1).eval().to(device)
    eigen_data = np.load(EIGENVECTORS_PATH)
    eigenvectors, mean = eigen_data["eigenvectors"], eigen_data["mean"]

    results = []

    for subject_idx in SUBJECT_IDX_LIST:
        cfg = {"neural_dataset": "nsd", "region": REGIONS[0], "subject_idx": subject_idx,
               "batchsize": 64, "num_workers": 8}
        _, stimuli = load_nsd_data(cfg)
        dataloader = _make_loader(stimuli, get_transform(ds_stats="imgnet"), batch=64, workers=8)
        activations, activation_ids = extract_fc2_activations(model, dataloader, device)

        aligned_data = {}
        for region in REGIONS:
            cfg["region"] = region
            neural_data, _ = load_nsd_data(cfg)
            aligned_data[region] = align_data(activations, activation_ids, neural_data)

        for n_pcs in N_PCS_LIST:
            for region in REGIONS:
                aligned_acts, aligned_neural = aligned_data[region]
                binary_codes = project_and_binarize(aligned_acts, eigenvectors, mean, n_pcs)
                neural_rsm = compute_rsm(torch.from_numpy(aligned_neural), correlation="Pearson")

                weighted_list = WEIGHTED if isinstance(WEIGHTED, list) else [WEIGHTED]
                corr_list = RSM_CORRELATION if isinstance(RSM_CORRELATION, list) else [RSM_CORRELATION]
                for weighted in weighted_list:
                    binary_rsm = compute_hamming_rsm(binary_codes, weighted=weighted)
                    for corr in corr_list:
                        score = _compare_rsms(binary_rsm, neural_rsm, correlation=corr)
                        results.append({
                            "subject_idx": subject_idx,
                            "n_pcs": n_pcs,
                            "region": region,
                            "weighted": weighted,
                            "correlation": corr,
                            "score": score,
                        })

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
    print(df.to_string(index=False))
    return df


if __name__ == "__main__":
    if not os.path.exists(EIGENVECTORS_PATH):
        print(f"Error: Eigenvectors not found: {EIGENVECTORS_PATH}")
        sys.exit(1)
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    run_analysis()
