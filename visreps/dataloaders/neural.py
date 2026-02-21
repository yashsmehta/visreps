import os, logging
from typing import Dict, Any, Tuple, List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import h5py

import visreps.utils as utils
from visreps.dataloaders.obj_cls import get_transform
logger = logging.getLogger(__name__)


# ───────────────────────── NSD ──────────────────────────
_NSD_REGION_MAP = {
    "early visual stream": "early",
    "ventral visual stream": "ventral",
    "V1": "V1",
    "V2": "V2",
    "V3": "V3",
    "hV4": "hV4",
    "FFA": "FFA",
    "PPA": "PPA",
}


def load_nsd_data(cfg: Dict) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
    """
    Load NSD fMRI responses with shared/unique train-test split.

    Loads consolidated nsd_data.pkl (all 8 subjects, streams ROIs) and splits
    stimuli into train (unique ~9,000) and test (shared 1,000) based on the
    shared1000 annotation.

    Args:
        cfg (Dict): Contains "region" and "subject_idx".

    Returns:
        (targets, stimuli):
            - targets: {"train": {stim_id: response}, "test": {stim_id: response}}
            - stimuli: {stim_id: np.ndarray} flat dict of all stimulus images
    """
    region_key = _NSD_REGION_MAP.get(cfg["region"], cfg["region"])
    subj = cfg["subject_idx"]

    root = utils.get_env_var("NSD_DATA_DIR")
    nsd = utils.load_pickle(os.path.join(root, "nsd_data.pkl"))

    shared_ids = nsd["shared_ids"]
    fmri_xr = nsd["data"][region_key][subj]

    stimulus_ids = [int(i) for i in fmri_xr.coords["stimulus"].values]

    # Split into train (unique) and test (shared)
    train_ids = [str(i) for i in stimulus_ids if i not in shared_ids]
    test_ids = [str(i) for i in stimulus_ids if i in shared_ids]

    targets = {
        "train": {i: fmri_xr.sel(stimulus=int(i)).values for i in train_ids},
        "test": {i: fmri_xr.sel(stimulus=int(i)).values for i in test_ids},
    }

    # Load images from HDF5 (flat dict covering all stimuli)
    hdf5_path = "/data/shared/datasets/allen2021.natural_scenes/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"
    images = {}
    with h5py.File(hdf5_path, "r") as f:
        imgBrick = f["imgBrick"]
        sorted_indices = np.sort(stimulus_ids)
        loaded_images = imgBrick[sorted_indices]
        for i, stim_id in enumerate(sorted_indices):
            images[str(stim_id)] = loaded_images[i]

    return targets, images


# ─────────────────── Lazy HDF5 Dict ────────────────────
class _LazyHdf5Dict:
    """Dict-like wrapper around an HDF5 dataset that reads images on demand.

    Avoids loading all images into RAM. Compatible with _StimuliDataset
    which accesses items via __getitem__.
    """

    def __init__(self, hdf5_path: str, dataset_name: str, indices):
        self._hdf5_path = hdf5_path
        self._dataset_name = dataset_name
        self._index_map = {str(idx): int(idx) for idx in indices}
        self._keys_sorted = sorted(self._index_map.keys(), key=lambda x: int(x))
        self._file = None

    def _open(self):
        if self._file is None:
            self._file = h5py.File(self._hdf5_path, "r")
        return self._file

    def __contains__(self, key):
        return str(key) in self._index_map

    def __len__(self):
        return len(self._index_map)

    def keys(self):
        return self._keys_sorted

    def __getitem__(self, key):
        key_str = str(key)
        if key_str not in self._index_map:
            raise KeyError(key)
        return self._open()[self._dataset_name][self._index_map[key_str]]

    def __del__(self):
        if self._file is not None:
            self._file.close()


# ─────────────────── NSD All Subjects ────────────────────
_NSD_SUBJECTS = list(range(8))


def load_all_nsd_data(cfg: Dict, subjects=None, regions=None) -> Dict:
    """
    Load NSD fMRI responses for requested subjects and regions.

    Args:
        cfg: Config dict.
        subjects: List of subject indices to load (default: all 8).
        regions: List of full region names to load (default: both streams).

    Returns:
        dict with keys:
            - "regions": list of full region names loaded
            - "subjects": list of subject indices loaded
            - "neural": {region: {subj: {"train": {sid: resp}, "test": {sid: resp}}}}
            - "stimuli": {str(stim_id): np.ndarray} union of all stimulus images
            - "shared_test_ids": sorted list of stimulus IDs shared across ALL subjects' test sets
    """
    subjects = subjects if subjects is not None else _NSD_SUBJECTS
    region_pairs = [(pkl_key, name) for name, pkl_key in _NSD_REGION_MAP.items()
                    if regions is None or name in regions]

    root = utils.get_env_var("NSD_DATA_DIR")
    nsd = utils.load_pickle(os.path.join(root, "nsd_data.pkl"))
    shared_ids = nsd["shared_ids"]

    neural = {}
    all_stimulus_ids = set()
    per_subject_test_ids = []

    for region_key, region_full in region_pairs:
        neural[region_full] = {}
        for subj in subjects:
            fmri_xr = nsd["data"][region_key][subj]
            stimulus_ids = [int(i) for i in fmri_xr.coords["stimulus"].values]
            all_stimulus_ids.update(stimulus_ids)

            train_ids = [str(i) for i in stimulus_ids if i not in shared_ids]
            test_ids = [str(i) for i in stimulus_ids if i in shared_ids]

            neural[region_full][subj] = {
                "train": {i: fmri_xr.sel(stimulus=int(i)).values for i in train_ids},
                "test": {i: fmri_xr.sel(stimulus=int(i)).values for i in test_ids},
            }

            # Collect test IDs per subject (first region only — same stimuli)
            if region_key == region_pairs[0][0]:
                per_subject_test_ids.append(set(test_ids))

    # shared_test_ids = intersection of all subjects' test sets
    shared_test_ids = sorted(set.intersection(*per_subject_test_ids), key=int)

    # Lazy HDF5 wrapper — reads images on demand, avoids loading 70k images (~36 GB) into RAM
    hdf5_path = "/data/shared/datasets/allen2021.natural_scenes/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"
    stimuli = _LazyHdf5Dict(hdf5_path, "imgBrick", all_stimulus_ids)

    region_names = [f for _, f in region_pairs]
    logger.info(
        f"Loaded NSD: {len(subjects)} subjects × {len(region_names)} regions, "
        f"{len(stimuli)} stimuli (lazy HDF5), {len(shared_test_ids)} shared test IDs"
    )

    return {
        "regions": region_names,
        "subjects": list(subjects),
        "neural": neural,
        "stimuli": stimuli,
        "shared_test_ids": shared_test_ids,
    }


# ─────────────── NSD Synthetic (Test Only) ────────────
def load_nsd_synthetic_test_data(cfg: Dict, subjects=None, regions=None) -> Dict:
    """
    Load NSD Synthetic test data only (220 shared stimuli).

    Layer selection is inherited from regular NSD evaluation, so no
    NSD train data is needed here.

    Returns:
        dict with keys:
            - "regions": list of region names
            - "subjects": list of subject indices
            - "neural": {region: {subj: {stim_name: response}}}
            - "stimuli": {stim_name: png_path}
            - "test_ids": sorted list of 220 synthetic stimulus names
    """
    subjects = subjects if subjects is not None else _NSD_SUBJECTS
    region_pairs = [(pkl_key, name) for name, pkl_key in _NSD_REGION_MAP.items()
                    if regions is None or name in regions]

    synth_root = utils.get_env_var("NSD_SYNTHETIC_DATA_DIR")
    synth = utils.load_pickle(os.path.join(synth_root, "nsd_synthetic_data.pkl"))
    shared_stimulus_names = synth["shared_stimulus_names"]

    neural = {}
    for region_key, region_full in region_pairs:
        neural[region_full] = {}
        for subj in subjects:
            synth_xr = synth["data"][region_key][subj]
            neural[region_full][subj] = {
                s: synth_xr.sel(stimulus=s).values
                for s in shared_stimulus_names
            }

    stimuli_dir = os.path.join(synth_root, "stimuli")
    stimuli = {name: os.path.join(stimuli_dir, f"{name}.png") for name in shared_stimulus_names}
    test_ids = shared_stimulus_names  # already sorted in pickle
    region_names = [f for _, f in region_pairs]

    logger.info(
        f"Loaded NSD Synthetic: {len(subjects)} subjects × {len(region_names)} regions, "
        f"{len(test_ids)} test stimuli"
    )

    return {
        "regions": region_names,
        "subjects": list(subjects),
        "neural": neural,
        "stimuli": stimuli,
        "test_ids": test_ids,
    }


# ─────────────────── NSD Synthetic ────────────────────
def load_nsd_synthetic_data(
    cfg: Dict,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load synthetic‑NSD fMRI responses and matching stimuli.

    Args:
        cfg (Dict): {"region": str, "subject_idx": int}

    Returns:
        (targets, stimuli):
            - targets: {stim_id: np.ndarray}
            - stimuli: {stim_id: PIL.Image.Image}
    """
    region, subj = cfg["region"], cfg["subject_idx"]
    root = utils.get_env_var("NSD_SYNTHETIC_DATA_DIR")

    fmri = utils.load_pickle(os.path.join(root, "fmri_responses.pkl"))[region][subj]
    images = {
        str(k): v
        for k, v in utils.load_pickle(os.path.join(root, f"stimuli_subject_{subj}.pkl")).items()
    }

    ids = {str(k) for k in fmri} & images.keys()

    return (
        {i: fmri[i] for i in ids},
        {i: images[i] for i in ids},
    )


# ────────────────────── CUSACK 2025 ─────────────────────
def load_cusack_data(cfg: Dict) -> Tuple[Dict[str, np.ndarray], Dict[str, str]]:
    """
    Load Cusack 2025 fMRI responses and stimulus images.

    Args:
        cfg (Dict): Contains "region" and optionally "age_group" ('2month' or '9month')

    Returns:
        (targets, stimuli):
            - targets: {stim_id: np.ndarray} - fMRI responses
            - stimuli: {stim_id: str} - paths to stimulus images
    """
    region = cfg["region"]
    age_group = cfg.get("age_group", "2month")  # Default to 2-month data

    # Load processed fMRI data
    fmri_path = os.path.join("datasets", "neural", "cusack2025", "fmri_responses.pkl")
    fmri_data = utils.load_pickle(fmri_path)

    targets = fmri_data[region][age_group]

    # Load stimulus images
    stimuli_dir = os.path.join("datasets", "neural", "cusack2025", "display_images")
    stimuli = {}

    for stim_id in targets.keys():
        img_path = os.path.join(stimuli_dir, f"{stim_id}.png")
        if os.path.exists(img_path):
            stimuli[stim_id] = img_path  # Store path for _StimuliDataset to handle
        else:
            raise FileNotFoundError(f"Stimulus image not found: {img_path}")

    return targets, stimuli


# ──────────────────────── THINGS ────────────────────────
def load_things_data() -> tuple[dict, dict[str, str]]:
    """
    Load THINGS behavioral dataset (concept embeddings + image IDs per concept).

    Expects cached pickle at datasets/neural/things/things_split.pkl
    (generate with: python scripts/preprocess_data/preprocess_things.py).

    Returns:
        targets: {
            "embeddings": {concept: np.ndarray(66,)},
            "image_ids":  {concept: [stimulus_id, ...]},
        }
        img_paths: {stimulus_id: path} for all images.
    """
    pkl_path = os.path.join("datasets", "neural", "things", "things_split.pkl")
    data = utils.load_pickle(pkl_path)

    targets = {
        "embeddings": data["embeddings"],
        "image_ids": data["image_ids"],
    }

    return targets, data["image_paths"]


# ──────────────────────── TVSD ────────────────────────
def _tvsd_things_image_path(sid: str, things_root: str) -> str | None:
    """Resolve a THINGS stimulus ID to its image path, or None if missing."""
    concept = "_".join(sid.split("_")[:-1])
    path = os.path.join(things_root, "images", "object_images", concept, f"{sid}.jpg")
    if os.path.exists(path):
        return path
    logger.warning(f"TVSD image not found: {path}")
    return None


def load_tvsd_data(cfg: Dict) -> tuple[dict, dict[str, str]]:
    """
    Load TVSD macaque MUA responses and corresponding THINGS image paths.

    Expects pickle with structure:
        data[region][subject_idx] = {"train": xr.DataArray, "test": xr.DataArray}
    Generate with: python scripts/preprocess_data/preprocess_tvsd.py

    Returns:
        targets: {"train": {sid: response}, "test": {sid: response}}
        img_paths: {sid: path} covering all stimuli (train + test).

    Config keys: region (V1/V4/IT), subject_idx (0=monkey F, 1=monkey N).
    """
    region, subj = cfg["region"], cfg["subject_idx"]
    fmri_path = os.path.join("datasets", "neural", "tvsd", "fmri_responses.pkl")
    splits = utils.load_pickle(fmri_path)[region][subj]

    things_root = os.path.join(
        os.environ.get("BONNER_DATASETS_HOME", os.path.expanduser("~/.cache/bonner-datasets")),
        "hebart2019.things",
    )

    targets = {}
    img_paths = {}
    for split_name, data_xr in splits.items():
        stim_ids = [str(s) for s in data_xr.coords["stimulus"].values]
        targets[split_name] = {
            sid: data_xr.sel(stimulus=sid).values for sid in stim_ids
        }
        for sid in stim_ids:
            if sid not in img_paths:
                p = _tvsd_things_image_path(sid, things_root)
                if p:
                    img_paths[sid] = p

    return targets, img_paths


# ─────────────────── TVSD All Subjects ────────────────────
_TVSD_REGIONS = ["V1", "V4", "IT"]
_TVSD_SUBJECTS = [0, 1]


def load_all_tvsd_data(cfg: Dict, subjects=None, regions=None) -> Dict:
    """
    Load TVSD macaque MUA responses for requested subjects and regions.

    Args:
        cfg: Config dict.
        subjects: List of subject indices to load (default: [0, 1]).
        regions: List of region names to load (default: ["V1", "V4", "IT"]).

    Returns:
        dict with keys:
            - "regions": list of region names loaded
            - "subjects": list of subject indices loaded
            - "neural": {region: {subj: {"train": {sid: resp}, "test": {sid: resp}}}}
            - "stimuli": {sid: image_path} union of all stimuli
            - "shared_test_ids": sorted list of test stimulus IDs shared across ALL subjects
    """
    subjects = subjects if subjects is not None else _TVSD_SUBJECTS
    regions_to_load = regions if regions is not None else _TVSD_REGIONS

    fmri_path = os.path.join("datasets", "neural", "tvsd", "fmri_responses.pkl")
    data = utils.load_pickle(fmri_path)

    things_root = os.path.join(
        os.environ.get("BONNER_DATASETS_HOME", os.path.expanduser("~/.cache/bonner-datasets")),
        "hebart2019.things",
    )

    neural = {}
    all_img_paths = {}
    per_subject_test_ids = []

    for region in regions_to_load:
        neural[region] = {}
        for subj in subjects:
            splits = data[region][subj]
            targets = {}
            for split_name, data_xr in splits.items():
                stim_ids = [str(s) for s in data_xr.coords["stimulus"].values]
                targets[split_name] = {
                    sid: data_xr.sel(stimulus=sid).values for sid in stim_ids
                }
                for sid in stim_ids:
                    if sid not in all_img_paths:
                        p = _tvsd_things_image_path(sid, things_root)
                        if p:
                            all_img_paths[sid] = p

            neural[region][subj] = targets

            # Collect test IDs per subject (first region only — same stimuli)
            if region == regions_to_load[0]:
                per_subject_test_ids.append(set(targets["test"].keys()))

    shared_test_ids = sorted(set.intersection(*per_subject_test_ids))

    logger.info(
        f"Loaded TVSD: {len(subjects)} subjects × {len(regions_to_load)} regions, "
        f"{len(all_img_paths)} stimuli, {len(shared_test_ids)} shared test IDs"
    )

    return {
        "regions": list(regions_to_load),
        "subjects": list(subjects),
        "neural": neural,
        "stimuli": all_img_paths,
        "shared_test_ids": shared_test_ids,
    }


# ─────────────────────── Dataset/Loader ───────────────────────
class _StimuliDataset(Dataset):
    """
    PyTorch Dataset for stimuli, supporting both file paths and in-memory image data.

    Args:
        stimuli (Dict[str, Any]): Mapping from stimulus ID to file path, np.ndarray, or PIL.Image.
        transform (callable): Transform to apply to each image.
    """

    def __init__(self, stimuli, transform):
        self.keys = sorted(stimuli.keys())
        # Store a reference — don't copy if stimuli is a lazy dict (e.g. _LazyHdf5Dict)
        self.stimuli = stimuli
        self.tr = transform or transforms.ToTensor()

    def __len__(self):
        """Return number of stimuli."""
        return len(self.keys)

    def _load_and_transform(self, data_or_path: Any, key: str):
        """
        Load and transform an image from a path, np.ndarray, or PIL.Image.
        Raises errors if image loading or transformation fails.
        """
        if isinstance(data_or_path, str):
            img = Image.open(data_or_path).convert("RGB")
        elif isinstance(data_or_path, np.ndarray):
            img = Image.fromarray(data_or_path.astype("uint8"), "RGB")
        elif isinstance(data_or_path, Image.Image):
            img = data_or_path.convert("RGB") if data_or_path.mode != "RGB" else data_or_path
        else:
            raise TypeError(f"Unsupported data type {type(data_or_path)} for key {key}")

        return self.tr(img)

    def __getitem__(self, idx):
        key = self.keys[idx]
        data_or_path = self.stimuli[key]
        transformed_img = self._load_and_transform(data_or_path, key)
        return transformed_img, key


def custom_collate_fn(
    batch: List[Tuple[torch.Tensor, str]]
) -> Tuple[torch.Tensor, List[str]]:
    imgs, keys = zip(*batch)
    return torch.stack(imgs), list(keys)


def _make_loader(stimuli, transform, batch, workers):
    return DataLoader(
        _StimuliDataset(stimuli, transform),
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        collate_fn=custom_collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
        prefetch_factor=2,
    )


def get_neural_loader(cfg: Dict) -> Tuple[Dict[str, Any], DataLoader]:
    """
    Returns (targets, dataloader) for the specified neural dataset.
    Supported datasets: 'nsd', 'things-behavior', 'nsd_synthetic', 'cusack', 'tvsd'.
    """
    dataset_type = cfg.get("neural_dataset")
    if dataset_type == "nsd":
        targets, stimuli = load_nsd_data(cfg)
    elif dataset_type == "things-behavior":
        targets, stimuli = load_things_data()
    elif dataset_type == "nsd_synthetic":
        targets, stimuli = load_nsd_synthetic_data(cfg)
    elif dataset_type == "cusack":
        targets, stimuli = load_cusack_data(cfg)
    elif dataset_type == "tvsd":
        targets, stimuli = load_tvsd_data(cfg)
    else:
        raise ValueError("neural_dataset must be 'nsd', 'things-behavior', 'nsd_synthetic', 'cusack', or 'tvsd'")

    transform = get_transform(ds_stats="imgnet")
    dataloader = _make_loader(
        stimuli,
        transform,
        cfg["batchsize"],
        cfg["num_workers"],
    )
    return targets, dataloader
