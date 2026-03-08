"""
Proof-of-principle evaluation on the Levels dataset (Muttenthaler et al., 2025).

Compares fine-grained (1000-class) vs coarse-grained models (4 PCA sources × 6
granularities) on:
  1. Odd-one-out accuracy — does the model predict the human-chosen odd-one-out?
  2. Uncertainty alignment — does model entropy correlate with human reaction time?
  3. Triplet RSA — Spearman r between model and human pairwise distances.

Usage (from project root):
    python experiments/levels_evaluation/evaluate_levels.py
"""

import os, sys, pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
from scipy.special import softmax
import torch
from omegaconf import OmegaConf

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from visreps.dataloaders.obj_cls import get_transform
from visreps.dataloaders.neural import _make_loader
import visreps.models.utils as mutils

# ── Config ──────────────────────────────────────────────
IMAGENET_DIR = "/data/shared/datasets/imagenet"
LEVELS_PKL = "/data/shared/datasets/muttenthaler2025.levels/processed_data/pruned_processed_dataset.pkl"
OUTPUT_DIR = ROOT / "experiments" / "levels_evaluation"
MIN_TRIALS = 300     # Minimum trials to consider a participant valid
TRIPLET_TYPES = ["between_class", "within_class", "class_border"]

# Best layer per model from NSD ventral stream RSA (averaged across subjects)
# Looked up from results.db: SELECT layer, AVG(score) ... GROUP BY cfg_id, checkpoint_dir, layer
BEST_LAYERS = {
    ("default", 1000): "fc1_pre",
    ("alexnet_pca", 2): "conv4_post", ("alexnet_pca", 4): "fc1_post",
    ("alexnet_pca", 8): "fc1_post",   ("alexnet_pca", 16): "fc1_post",
    ("alexnet_pca", 32): "fc1_post",  ("alexnet_pca", 64): "fc1_post",
    ("clip_pca", 2): "conv4_post",    ("clip_pca", 4): "fc1_post",
    ("clip_pca", 8): "fc1_post",      ("clip_pca", 16): "fc1_post",
    ("clip_pca", 32): "fc2_pre",      ("clip_pca", 64): "fc2_pre",
    ("dino_pca", 2): "fc1_pre",       ("dino_pca", 4): "fc1_post",
    ("dino_pca", 8): "fc1_post",      ("dino_pca", 16): "fc1_post",
    ("dino_pca", 32): "fc1_post",     ("dino_pca", 64): "fc2_pre",
    ("vit_pca", 2): "conv4_pre",      ("vit_pca", 4): "fc1_post",
    ("vit_pca", 8): "fc1_post",       ("vit_pca", 16): "fc1_post",
    ("vit_pca", 32): "fc2_pre",       ("vit_pca", 64): "fc1_pre",
}

PCA_SOURCES = ["alexnet_pca", "clip_pca", "dino_pca", "vit_pca"]
COARSENESS_LEVELS = [2, 4, 8, 16, 32, 64]

def build_models():
    """Build model configs: 1 fine-grained + 24 coarse-grained."""
    models = {
        "cfg1000": {
            "checkpoint_dir": "/data/ymehta3/default",
            "cfg_id": 1000, "seed": 1,
            "checkpoint_model": "checkpoint_epoch_20.pth",
            "layer": BEST_LAYERS[("default", 1000)],
        }
    }
    for src in PCA_SOURCES:
        short = src.replace("_pca", "")
        for n in COARSENESS_LEVELS:
            models[f"cfg{n}_{short}"] = {
                "checkpoint_dir": f"/data/ymehta3/{src}",
                "cfg_id": n, "seed": 1,
                "checkpoint_model": "checkpoint_epoch_20.pth",
                "layer": BEST_LAYERS[(src, n)],
            }
    return models


# ── Data loading ────────────────────────────────────────
def load_levels_data():
    """Load Levels dataset and filter to triplets with all 3 images available."""
    with open(LEVELS_PKL, "rb") as f:
        data = pickle.load(f)

    # Collect all unique triplets with majority vote and mean RT
    triplet_info = {}  # (img1, img2, img3) sorted -> {votes, rts, triplet_type, images}
    for pid, trials in data.items():
        if len(trials) < MIN_TRIALS:
            continue
        for t in trials:
            key = tuple(sorted([t["image1Path"], t["image2Path"], t["image3Path"]]))
            if key not in triplet_info:
                triplet_info[key] = {"votes": [], "rts": [], "triplet_type": t["triplet_type"],
                                     "images": [t["image1Path"], t["image2Path"], t["image3Path"]]}
            triplet_info[key]["votes"].append(t["selected_image"])
            triplet_info[key]["rts"].append(t["rt"])

    # Build set of available images upfront (one listdir per synset, not per image)
    all_images = {img for key in triplet_info for img in key}
    synset_contents = {}
    for img in all_images:
        synset = img.split("_")[0]
        if synset not in synset_contents:
            synset_dir = os.path.join(IMAGENET_DIR, synset)
            synset_contents[synset] = set(os.listdir(synset_dir)) if os.path.isdir(synset_dir) else set()
    available = {img for img in all_images if img in synset_contents.get(img.split("_")[0], set())}

    # Filter to triplets where all 3 images exist
    triplets = []
    for key, info in triplet_info.items():
        if not all(img in available for img in key):
            continue
        vote_counts = Counter(info["votes"])
        total_votes = sum(vote_counts.values())
        majority = vote_counts.most_common(1)[0][0]
        vote_probs = {img: vote_counts.get(img, 0) / total_votes
                      for img in info["images"]}
        paths = {img: os.path.join(IMAGENET_DIR, img.split("_")[0], img) for img in key}
        triplets.append({
            "images": info["images"],
            "paths": paths,
            "human_choice": majority,
            "vote_probs": vote_probs,
            "mean_rt": np.mean(info["rts"]),
            "triplet_type": info["triplet_type"],
        })

    # Collect unique images for the dataloader
    all_stimuli = {}
    for t in triplets:
        all_stimuli.update(t["paths"])

    print(f"Loaded {len(triplets)} complete triplets ({len(all_stimuli)} unique images)")
    for tt in TRIPLET_TYPES:
        print(f"  {tt}: {sum(1 for t in triplets if t['triplet_type'] == tt)}")
    return triplets, all_stimuli


# ── Feature extraction ──────────────────────────────────
def make_dataloader(stimuli, batchsize=64, num_workers=8):
    """Create a reusable dataloader for the Levels stimuli."""
    transform = get_transform(ds_stats="imgnet")
    return _make_loader(stimuli, transform, batchsize, num_workers)


def extract_features(model_cfg, dl, layer):
    """Load model, extract layer activations (pre-normalized) for all stimuli."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = OmegaConf.create({
        "load_model_from": "checkpoint",
        "model_class": "custom_model",
        "model_name": "CustomCNN",
        "return_nodes": ["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2"],
        "extract_pre_and_post": True,
        **model_cfg,
    })
    seed_letter = mutils.get_seed_letter(cfg.seed)
    train_cfg = OmegaConf.load(f"{cfg.checkpoint_dir}/cfg{cfg.cfg_id}{seed_letter}/config.json")
    cfg = OmegaConf.merge(train_cfg, cfg)

    model = mutils.load_model(cfg, dev)
    model = mutils.configure_feature_extractor(cfg, model)

    acts, ids = mutils.extract_single_layer(model, dl, dev, layer)

    # Pre-normalize: center + L2-normalize so Pearson distance = 1 - dot product
    acts = acts.float()
    acts = acts - acts.mean(dim=1, keepdim=True)
    acts = acts / (acts.norm(dim=1, keepdim=True) + 1e-12)
    acts = acts.cpu()

    features = {img_id: acts[i] for i, img_id in enumerate(ids)}

    del model, dl, acts
    torch.cuda.empty_cache()
    return features


# ── Metrics ─────────────────────────────────────────────
def compute_triplet_distances(features, images):
    """Compute 3 pairwise Pearson distances for a triplet (pre-normalized: 1 - dot)."""
    a, b, c = features[images[0]], features[images[1]], features[images[2]]
    return np.array([1.0 - (a @ b).item(), 1.0 - (a @ c).item(), 1.0 - (b @ c).item()])


def predict_odd_one_out(dists, images):
    """Predict odd-one-out from precomputed distances [d(A,B), d(A,C), d(B,C)]."""
    dab, dac, dbc = dists
    scores = {
        images[0]: (dab + dac) / 2,
        images[1]: (dab + dbc) / 2,
        images[2]: (dac + dbc) / 2,
    }
    return max(scores, key=scores.get)


# ── Main evaluation ─────────────────────────────────────
def evaluate_model(model_name, features, triplets):
    """Evaluate a single model: accuracy, uncertainty, and triplet RSA."""
    results = []
    rsa_by_type = defaultdict(lambda: {"model": [], "human": []})

    for t in triplets:
        imgs = t["images"]
        dists = compute_triplet_distances(features, imgs)

        # Odd-one-out accuracy + uncertainty
        pred = predict_odd_one_out(dists, imgs)
        entropy = scipy.stats.entropy(softmax(dists))
        results.append({
            "model": model_name,
            "triplet_type": t["triplet_type"],
            "correct": int(pred == t["human_choice"]),
            "model_entropy": entropy,
            "human_rt": t["mean_rt"],
        })

        # Triplet RSA: human distance = 1 - P(third image is odd-one-out)
        vp = t["vote_probs"]
        human_dists = [1.0 - vp[imgs[2]], 1.0 - vp[imgs[1]], 1.0 - vp[imgs[0]]]
        tt = t["triplet_type"]
        for d_m, d_h in zip(dists, human_dists):
            rsa_by_type[tt]["model"].append(d_m)
            rsa_by_type[tt]["human"].append(d_h)
            rsa_by_type["overall"]["model"].append(d_m)
            rsa_by_type["overall"]["human"].append(d_h)

    rsa = {}
    for tt, data in rsa_by_type.items():
        r, p = scipy.stats.spearmanr(data["model"], data["human"])
        rsa[tt] = (round(r, 4), round(p, 6))

    return results, rsa


def build_summary(df):
    """Build summary DataFrame with accuracy and uncertainty alignment."""
    rows = []
    for model in df["model"].unique():
        mdf = df[df["model"] == model]
        for tt in TRIPLET_TYPES + ["overall"]:
            sub = mdf if tt == "overall" else mdf[mdf["triplet_type"] == tt]
            r, p = scipy.stats.spearmanr(sub["model_entropy"], sub["human_rt"])
            rows.append({
                "model": model, "triplet_type": tt,
                "accuracy": round(sub["correct"].mean(), 4),
                "uncertainty_r": round(r, 4), "uncertainty_p": round(p, 6),
                "n_triplets": len(sub),
            })
    return pd.DataFrame(rows)


def main():
    triplets, stimuli = load_levels_data()

    models = build_models()
    dl = make_dataloader(stimuli)

    all_results = []
    all_rsa = []
    for model_name, model_cfg in models.items():
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*60}")
        layer = model_cfg["layer"]
        cfg_for_model = {k: v for k, v in model_cfg.items() if k != "layer"}
        features = extract_features(cfg_for_model, dl, layer)
        results, rsa = evaluate_model(model_name, features, triplets)
        all_results.extend(results)
        for tt, (r, p) in rsa.items():
            all_rsa.append({"model": model_name, "triplet_type": tt, "rsa_r": r, "rsa_p": p})
        del features

    df = pd.DataFrame(all_results)
    rsa_df = pd.DataFrame(all_rsa)
    summary = build_summary(df).merge(rsa_df, on=["model", "triplet_type"], how="left")

    # ── Print results ───────────────────────────────────
    print(f"\n{'='*60}")
    print("ODD-ONE-OUT ACCURACY")
    print(f"{'='*60}")
    print(summary.pivot(index="model", columns="triplet_type", values="accuracy").to_string())

    print(f"\n{'='*60}")
    print("UNCERTAINTY ALIGNMENT (Spearman r: model entropy vs human RT)")
    print(f"{'='*60}")
    print(summary.pivot(index="model", columns="triplet_type", values="uncertainty_r").to_string())

    print(f"\n{'='*60}")
    print("TRIPLET RSA (Spearman r: model distances vs human behavioral distances)")
    print(f"{'='*60}")
    print(summary.pivot(index="model", columns="triplet_type", values="rsa_r").to_string())

    # ── Save ────────────────────────────────────────────
    df.to_csv(OUTPUT_DIR / "levels_results.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "levels_summary.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
