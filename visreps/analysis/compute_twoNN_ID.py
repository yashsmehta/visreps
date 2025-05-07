"""
twoNN_id_pipeline.py
────────────────────
Compute intrinsic dimensionality (Facco Two‑NN) for each layer in an .npz
(model activations).  Designed for big matrices (≈50 k × 4096).

Key changes
* clean  √  duplicate/NaN rows before any work
* fast   √  optional FAISS (GPU/CPU) fallback to scikit‑learn
* correct√  rebuild NN index inside every decimation subsample
* lean   √  single‑pass CSV append; minimal branching / warnings
"""

from __future__ import annotations
import argparse, csv, os, re, warnings, numpy as np

# ───────────────────────────────────────────────────────── NN back‑end ──
try:                                  # FAISS is ~10× faster than sklearn
    import faiss
    _FAISS_OK = True
except ModuleNotFoundError:
    from sklearn.neighbors import NearestNeighbors
    _FAISS_OK = False


# ───────────────────────────────────────────────────────── core routine ──
def twoNN_id(X: np.ndarray,
             decimate: list[int] = (1, 2, 5, 10),
             rng: np.random.Generator | None = None,
             n_jobs: int = -1) -> tuple[float, dict[int, float]]:
    """Return (ID@k=1 , {k:ID})."""
    X = np.asarray(X, dtype=np.float32)
    ok = np.isfinite(X).all(axis=1)         # drop rows w/ NaN/Inf once
    X = X[ok]
    N = len(X)
    if N < 3:
        return np.nan, {k: np.nan for k in decimate}

    rng = rng or np.random.default_rng()
    id_by_k: dict[int, float] = {}

    def _fit_knn(A: np.ndarray):
        if _FAISS_OK:
            idx = faiss.IndexFlatL2(A.shape[1])
            idx.add(A)
            return idx
        return NearestNeighbors(
            n_neighbors=min(3, len(A)-1),
            metric='euclidean',
            n_jobs=n_jobs).fit(A)

    for k in sorted(set(decimate)):
        m = N // k
        if m < 3:
            id_by_k[k] = np.nan
            continue
        idx = slice(None) if k == 1 else rng.choice(N, m, replace=False)
        A = X[idx]

        knn = _fit_knn(A)
        if _FAISS_OK:
            d, I = knn.search(A, 3)               # (m,3) squared dists
            d = np.sqrt(d)                        # need Euclidean
        else:
            d, I = knn.kneighbors(A, 3, return_distance=True)

        # remove self (index 0) duplicates & ties between NN1/NN2
        good = (I[:,1] != I[:,0]) & (I[:,2] != I[:,0]) & (I[:,1] != I[:,2])
        r1, r2 = d[good,1], d[good,2]
        keep = (r1 > 0) & (r2 > 0)
        if not keep.any():
            id_by_k[k] = np.nan
            continue

        mu = r2[keep] / r1[keep]
        id_by_k[k] = 1.0 / np.mean(np.log(mu))

    return id_by_k.get(1, np.nan), id_by_k


# ───────────────────────────────────────────────────── layer‑wise helper ──
def intrinsic_dim_layer(mat: np.ndarray,
                         decimate: list[int],
                         n_jobs: int) -> tuple[float, float]:
    id1, id_dict = twoNN_id(mat, decimate, n_jobs=n_jobs)
    if np.isnan(id1):
        return np.nan, np.nan
    dev = [abs(v - id1) / id1 for k, v in id_dict.items()
           if k > 1 and np.isfinite(v)]
    return id1, (max(dev) * 100 if dev else 0.0)


# ─────────────────────────────────────────────── .npz → CSV processing ──
def process_file(path: str, decimate: list[int], n_jobs: int,
                 csv_name: str = "twoNN_intrinsic_dimensionality.csv"):
    with np.load(path, allow_pickle=True) as data:
        layers = [k for k in data.files if k.startswith(("conv", "fc"))]
        if not layers:
            print(f"No conv/fc arrays in {os.path.basename(path)}"); return

        epoch = re.search(r'_epoch_(\d+)', path)
        epoch = int(epoch.group(1)) if epoch else "NA"

        rows = []
        for k in layers:
            arr = np.asarray(data[k])
            if arr.ndim == 1:
                arr = arr[:, None]
            elif arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)

            id1, var = intrinsic_dim_layer(arr, decimate, n_jobs)
            if np.isnan(id1): continue
            rows.append({"Epoch": epoch,
                         "Layer": k,
                         "Intrinsic_Dimensionality": f"{id1:.4f}",
                         "Max_Variation_%": f"{var:.1f}"})

            print(f"{k:20s}  ID={id1:7.2f}  Δ={var:5.1f}%")

    if not rows: return
    out_csv = os.path.join(os.path.dirname(path), csv_name)
    head = not os.path.exists(out_csv)
    with open(out_csv, "a", newline="") as f:
        w = csv.DictWriter(f, rows[0].keys())
        if head: w.writeheader()
        w.writerows(rows)


# ──────────────────────────────────────────────────────────── CLI ──
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="model_checkpoints/imagenet_cnn/cfg1/model_representations/model_reps_epoch_25.npz")
    ap.add_argument("--decimate", default="1,2,5,10")
    ap.add_argument("--n_jobs", type=int, default=-1)
    args = ap.parse_args()

    decim = sorted({int(x) for x in args.decimate.split(",") if int(x) > 0} | {1})
    process_file(args.input, decim, args.n_jobs)