"""Run the ImageNet-1k fine-class manifold SNR experiment.

All models are evaluated on the same 100 fixed 50-class panels. Training labels
define model conditions only; manifolds are always grouped by original
ImageNet-1k validation labels.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from experiments.manifold_analysis.manifold_snr import (
    empirical_nearest_prototype_error,
    fit_manifold_geometry,
    fit_pairwise_geometry,
    manifold_snr,
    snr_from_geometry,
)
from visreps.models import custom_model


CONDITIONS = (2, 4, 8, 16, 32, 64, 1000)
SEEDS = ((1, "a"), (2, "b"), (3, "c"))
SHOT_COUNTS = (1, 5, 10)
COARSE_CHECKPOINT_ROOT = Path("/data/ymehta3/clip_pca")
FINE_CHECKPOINT_ROOT = Path("/data/ymehta3/default")
DEFAULT_OUTPUT_DIR = Path("experiments/manifold_analysis/snr_results")
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def checkpoint_path(n_classes: int, seed_letter: str) -> Path:
    root = FINE_CHECKPOINT_ROOT if n_classes == 1000 else COARSE_CHECKPOINT_ROOT
    return root / f"cfg{n_classes}{seed_letter}" / "checkpoint_epoch_20.pth"


def load_model(n_classes: int, seed_letter: str, device: torch.device):
    """Load a pickled CustomCNN checkpoint and put it in inference mode."""
    sys.modules["visreps.models.custom_cnn"] = custom_model
    path = checkpoint_path(n_classes, seed_letter)
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    return checkpoint["model"].eval().to(device), path


def image_panel(imagenet_dir: Path, images_per_class: int) -> tuple[list[str], list[list[str]]]:
    """Return every original ImageNet class and a stable list of its images."""
    if (imagenet_dir / "val").is_dir():
        imagenet_dir = imagenet_dir / "val"
    dataset = ImageFolder(imagenet_dir)
    by_class: list[list[str]] = [[] for _ in dataset.classes]
    for path, class_index in dataset.samples:
        by_class[class_index].append(path)
    if len(dataset.classes) != 1000:
        raise ValueError(f"expected 1,000 ImageNet classes, found {len(dataset.classes)}")
    wrong_size = [
        (dataset.classes[i], len(paths))
        for i, paths in enumerate(by_class)
        if len(paths) != images_per_class
    ]
    if wrong_size:
        raise ValueError(
            f"expected exactly {images_per_class} validation images per class, but "
            f"{len(wrong_size)} classes differ; first: {wrong_size[0]}. "
            "Point --imagenet-dir at the ImageNet-1k validation split, not training data."
        )
    return dataset.classes, [sorted(paths) for paths in by_class]


def manifest_panel(
    manifest_path: Path, images_per_class: int
) -> tuple[list[str], list[list[str]]]:
    """Load and validate the persisted historical 20% holdout selection."""
    manifest = json.loads(manifest_path.read_text())
    category_map = manifest.get("categories")
    if not isinstance(category_map, dict) or len(category_map) != 1000:
        raise ValueError("held-out manifest must contain a mapping of 1,000 classes")
    classes = sorted(category_map)
    paths = [category_map[category] for category in classes]
    wrong_size = [
        (category, len(category_paths))
        for category, category_paths in zip(classes, paths, strict=True)
        if len(category_paths) != images_per_class
    ]
    if wrong_size:
        raise ValueError(
            f"expected {images_per_class} selected images per class; first mismatch: "
            f"{wrong_size[0]}"
        )
    flat = [path for category_paths in paths for path in category_paths]
    if len(set(flat)) != len(flat):
        raise ValueError("held-out manifest contains duplicate image paths")
    missing = [path for path in flat if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} manifest images are missing; first: {missing[0]}")
    return classes, paths


def make_sampling_plan(
    n_classes: int,
    *,
    repetitions: int,
    classes_per_repetition: int,
    images_per_class: int,
    seed: int,
) -> dict[str, object]:
    """Create class subsets and a common 25-image robustness subset."""
    if classes_per_repetition > n_classes:
        raise ValueError("classes per repetition exceeds available classes")
    rng = np.random.default_rng(seed)
    subsets = [
        np.sort(rng.choice(n_classes, classes_per_repetition, replace=False)).tolist()
        for _ in range(repetitions)
    ]
    image_subset_25 = np.sort(
        rng.choice(images_per_class, min(25, images_per_class), replace=False)
    ).tolist()
    return {
        "seed": seed,
        "repetitions": repetitions,
        "classes_per_repetition": classes_per_repetition,
        "images_per_class": images_per_class,
        "class_subsets": subsets,
        "image_subset_25": image_subset_25,
    }


@torch.inference_mode()
def extract_fc7(
    model,
    image_paths: list[list[str]],
    *,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Extract the unprojected 4,096-D FC7 activation after ReLU."""
    transform = transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    flat_paths = [path for paths in image_paths for path in paths]
    values = []
    for start in tqdm(range(0, len(flat_paths), batch_size), desc="FC7 batches"):
        images = [
            transform(Image.open(path).convert("RGB"))
            for path in flat_paths[start : start + batch_size]
        ]
        x = torch.stack(images).to(device)
        x = model.features(x)
        x = model.adaptive_pool(x).flatten(1)
        x = model.classifier[:8](x)
        values.append(x.float().cpu())
    n_classes, n_images = len(image_paths), len(image_paths[0])
    return (
        torch.cat(values)
        .reshape(n_classes, n_images, 4096)
        .permute(0, 2, 1)
        .numpy()
    )


def off_diagonal_mean(value: np.ndarray) -> float:
    return float(np.nanmean(value))


def sample_std(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def correlation(x: np.ndarray, y: np.ndarray, *, rank: bool = False) -> float | None:
    if len(x) < 2 or np.all(x == x[0]) or np.all(y == y[0]):
        return None
    statistic = spearmanr(x, y).statistic if rank else pearsonr(x, y).statistic
    return float(statistic)


def summarize_snr(result: dict[str, object]) -> dict[str, float]:
    """Collapse the pair matrices while retaining every requested contribution."""
    return {
        "mean_snr": float(result["mean"]),
        "predicted_error": off_diagonal_mean(result["predicted_error"]),
        "signal": off_diagonal_mean(result["signal"]),
        "bias": off_diagonal_mean(result["bias"]),
        "dimension": float(np.mean(result["dimension"])),
        "dimension_noise": off_diagonal_mean(result["dimension_noise"]),
        "signal_noise_self": off_diagonal_mean(result["signal_noise_self"]),
        "signal_noise_other": off_diagonal_mean(result["signal_noise_other"]),
        "noise_noise": off_diagonal_mean(result["noise_noise"]),
        "numerator": off_diagonal_mean(result["numerator"]),
        "denominator": off_diagonal_mean(result["denominator"]),
    }


def analyze_model(
    manifolds: np.ndarray,
    plan: dict[str, object],
    *,
    model_seed: int,
    empirical_pairs: int,
    empirical_trials: int,
    pair_output: Path,
) -> dict[str, object]:
    """Analyze fixed panels, sensitivity shots, sample-size stability, and error."""
    repetitions: list[dict[str, object]] = []
    pair_arrays: dict[str, np.ndarray] = {}
    subsets = plan["class_subsets"]
    image_subset_25 = np.asarray(plan["image_subset_25"])
    for repetition, class_indices_list in enumerate(tqdm(subsets, desc="SNR panels")):
        class_indices = np.asarray(class_indices_list)
        panel = manifolds[class_indices]
        entry: dict[str, object] = {
            "repetition": repetition,
            "class_indices": class_indices.tolist(),
            "shots": {},
        }
        geometry = fit_manifold_geometry(panel)
        pair_geometry = fit_pairwise_geometry(geometry)
        primary = None
        for n_shots in SHOT_COUNTS:
            result = snr_from_geometry(
                geometry, n_shots=n_shots, pair_geometry=pair_geometry
            )
            entry["shots"][str(n_shots)] = summarize_snr(result)
            pair_arrays[f"rep{repetition:03d}_m{n_shots}_snr"] = result["pairwise"].astype(np.float32)
            pair_arrays[f"rep{repetition:03d}_m{n_shots}_predicted_error"] = result[
                "predicted_error"
            ].astype(np.float32)
            if n_shots == 5:
                primary = result
                for name in (
                    "signal",
                    "bias",
                    "dimension_noise",
                    "signal_noise_self",
                    "signal_noise_other",
                    "noise_noise",
                ):
                    pair_arrays[f"rep{repetition:03d}_m5_{name}"] = result[name].astype(np.float32)

        result_25 = manifold_snr(panel[:, :, image_subset_25], n_shots=5)
        entry["images_per_class_25"] = summarize_snr(result_25)
        if repetition == 0 and empirical_pairs:
            rng = np.random.default_rng(int(plan["seed"]) + model_seed)
            candidates = np.argwhere(~np.eye(len(panel), dtype=bool))
            chosen = candidates[
                rng.choice(len(candidates), min(empirical_pairs, len(candidates)), replace=False)
            ]
            empirical = empirical_nearest_prototype_error(
                panel,
                chosen,
                n_shots=5,
                n_trials=empirical_trials,
                seed=int(plan["seed"]) + 10_000 + model_seed,
            )
            predicted = primary["predicted_error"][chosen[:, 0], chosen[:, 1]]
            entry["empirical_validation"] = {
                "pairs": chosen.tolist(),
                "predicted_error": predicted.tolist(),
                "empirical_error": empirical.tolist(),
                "pearson_r": correlation(predicted, empirical),
                "spearman_r": correlation(predicted, empirical, rank=True),
                "mean_absolute_error": float(np.mean(np.abs(predicted - empirical))),
                "n_trials": empirical_trials,
            }
        repetitions.append(entry)

    np.savez_compressed(pair_output, **pair_arrays)
    metrics = (
        "mean_snr",
        "predicted_error",
        "signal",
        "bias",
        "dimension",
        "dimension_noise",
        "signal_noise_self",
        "signal_noise_other",
        "noise_noise",
        "numerator",
        "denominator",
    )
    aggregate = {}
    for n_shots in SHOT_COUNTS:
        shot_entries = [rep["shots"][str(n_shots)] for rep in repetitions]
        aggregate[str(n_shots)] = {
            metric: {
                "mean": float(np.mean([entry[metric] for entry in shot_entries])),
                "std": sample_std([entry[metric] for entry in shot_entries]),
            }
            for metric in metrics
        }
    snr_50 = np.asarray([rep["shots"]["5"]["mean_snr"] for rep in repetitions])
    snr_25 = np.asarray([rep["images_per_class_25"]["mean_snr"] for rep in repetitions])
    return {
        "aggregate": aggregate,
        "sample_size_stability": {
            "mean_snr_50": float(snr_50.mean()),
            "mean_snr_25": float(snr_25.mean()),
            "pearson_r_across_repetitions": correlation(snr_50, snr_25),
            "mean_relative_change": float(np.mean((snr_25 - snr_50) / snr_50)),
        },
        "repetitions": repetitions,
        "pairwise_file": str(pair_output),
    }


def add_retention(results: dict[str, object]) -> None:
    """Add retention and component deltas versus the seed-matched baseline."""
    models = results["models"]
    for seed_number, seed_letter in SEEDS:
        baseline = models.get(f"cfg1000{seed_letter}")
        if baseline is None:
            continue
        baseline_values = np.asarray(
            [rep["shots"]["5"]["mean_snr"] for rep in baseline["repetitions"]]
        )
        for n_classes in CONDITIONS:
            entry = models.get(f"cfg{n_classes}{seed_letter}")
            if entry is None:
                continue
            values = np.asarray([rep["shots"]["5"]["mean_snr"] for rep in entry["repetitions"]])
            retention = values / baseline_values
            entry["snr_retention"] = {
                "per_repetition": retention.tolist(),
                "mean": float(retention.mean()),
                "std": sample_std(retention.tolist()),
            }
            component_names = (
                "signal",
                "bias",
                "dimension",
                "dimension_noise",
                "signal_noise_self",
                "signal_noise_other",
                "noise_noise",
            )
            entry["snr_difference_decomposition"] = {}
            for component in component_names:
                component_values = np.asarray(
                    [rep["shots"]["5"][component] for rep in entry["repetitions"]]
                )
                baseline_components = np.asarray(
                    [rep["shots"]["5"][component] for rep in baseline["repetitions"]]
                )
                delta = component_values - baseline_components
                entry["snr_difference_decomposition"][component] = {
                    "per_repetition_delta": delta.tolist(),
                    "mean_delta": float(delta.mean()),
                    "std_delta": sample_std(delta.tolist()),
                }


def plot_retention(results: dict[str, object], path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.5))
    for _, seed_letter in SEEDS:
        xs, ys = [], []
        for n_classes in CONDITIONS:
            entry = results["models"].get(f"cfg{n_classes}{seed_letter}")
            if entry and "snr_retention" in entry:
                xs.append(n_classes)
                ys.append(entry["snr_retention"]["mean"])
        if xs:
            ax.plot(xs, ys, "o-", alpha=0.75, label=f"seed {seed_letter}")
    ax.axhline(1, color="black", linestyle="--", linewidth=1, label="1,000-way baseline")
    ax.set_xscale("log", base=2)
    ax.set_xticks(CONDITIONS, labels=[str(value) for value in CONDITIONS])
    ax.set_xlabel("Number of CLIP-derived training labels (K)")
    ax.set_ylabel("Five-shot SNR retention")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def save_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("experiments/manifold_analysis/heldout_dataset/manifest.json"),
        help="50-images-per-class manifest sampled from the historical 20%% holdout",
    )
    parser.add_argument(
        "--imagenet-dir",
        type=Path,
        default=None,
        help="optional class-directory dataset used instead of --manifest",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--conditions", type=int, nargs="+", default=list(CONDITIONS))
    parser.add_argument("--seed-letters", nargs="+", default=[letter for _, letter in SEEDS])
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--classes-per-repetition", type=int, default=50)
    parser.add_argument("--images-per-class", type=int, default=50)
    parser.add_argument("--sampling-seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--empirical-pairs", type=int, default=100)
    parser.add_argument("--empirical-trials", type=int, default=100)
    parser.add_argument("--refresh-activations", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    invalid = sorted(set(args.conditions) - set(CONDITIONS))
    if invalid:
        raise SystemExit(f"unsupported conditions: {invalid}")
    seed_map = {letter: number for number, letter in SEEDS}
    if set(args.seed_letters) - set(seed_map):
        raise SystemExit("seed letters must be selected from a, b, c")
    if args.images_per_class < 25:
        raise SystemExit("--images-per-class must be at least 25")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "sampling_plan.json"
    if args.imagenet_dir is not None:
        classes, paths = image_panel(args.imagenet_dir, args.images_per_class)
        data_source = str(args.imagenet_dir)
    else:
        classes, paths = manifest_panel(args.manifest, args.images_per_class)
        data_source = str(args.manifest)
    if manifest_path.exists():
        plan = json.loads(manifest_path.read_text())
        expected = (args.repetitions, args.classes_per_repetition, args.images_per_class)
        actual = (
            plan["repetitions"],
            plan["classes_per_repetition"],
            plan["images_per_class"],
        )
        if actual != expected:
            raise ValueError(
                f"existing sampling plan parameters {actual} do not match requested {expected}; "
                "use a new output directory"
            )
        classes, paths = plan["class_names"], plan["image_paths"]
    else:
        plan = make_sampling_plan(
            len(classes),
            repetitions=args.repetitions,
            classes_per_repetition=args.classes_per_repetition,
            images_per_class=args.images_per_class,
            seed=args.sampling_seed,
        )
        plan["class_names"] = classes
        plan["image_paths"] = paths
        plan["data_source"] = data_source
        save_json(manifest_path, plan)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    use_cuda = args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())
    device = torch.device("cuda:0" if use_cuda else "cpu")
    results_path = args.output_dir / "results.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {
        "experiment": {
            "representation": "CustomCNN FC7 (classifier second hidden layer), post-ReLU",
            "feature_dimension": 4096,
            "projection": None,
            "evaluation_labels": "original ImageNet-1k class labels",
            "training_conditions": list(CONDITIONS),
            "shot_counts": list(SHOT_COUNTS),
            "primary_shots": 5,
            "sampling_plan": str(manifest_path),
        },
        "models": {},
    }

    jobs = [
        (n_classes, seed_map[letter], letter)
        for n_classes in args.conditions
        for letter in args.seed_letters
    ]
    for n_classes, seed_number, seed_letter in jobs:
        model_key = f"cfg{n_classes}{seed_letter}"
        if model_key in results["models"]:
            print(f"Skipping completed {model_key}")
            continue
        activation_path = args.output_dir / f"{model_key}_fc7.npz"
        if activation_path.exists() and not args.refresh_activations:
            with np.load(activation_path) as saved:
                manifolds = saved["fc7"]
            model_path = checkpoint_path(n_classes, seed_letter)
        else:
            model, model_path = load_model(n_classes, seed_letter, device)
            manifolds = extract_fc7(model, paths, batch_size=args.batch_size, device=device)
            np.savez_compressed(activation_path, fc7=manifolds.astype(np.float32))
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        model_result = analyze_model(
            manifolds,
            plan,
            model_seed=seed_number,
            empirical_pairs=args.empirical_pairs,
            empirical_trials=args.empirical_trials,
            pair_output=args.output_dir / f"{model_key}_pairwise.npz",
        )
        model_result.update(
            {
                "n_training_classes": n_classes,
                "seed": seed_number,
                "seed_letter": seed_letter,
                "checkpoint": str(model_path),
                "activation_file": str(activation_path),
            }
        )
        results["models"][model_key] = model_result
        add_retention(results)
        save_json(results_path, results)
        plot_retention(results, args.output_dir / "snr_retention.png")

    print(f"Saved results to {results_path}")


if __name__ == "__main__":
    main()
