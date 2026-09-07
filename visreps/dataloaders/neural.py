import os, logging
from typing import Dict, Any, Tuple, List
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import h5py

import visreps.utils as utils
from visreps.dataloaders.obj_cls import get_transform, DS_MEAN, DS_STD
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

    Loads consolidated nsd_data.pkl (all 8 subjects, NCSNR-filtered at
    preprocessing time) and splits stimuli into train (unique ~9,000) and test
    (shared 1,000) based on the shared1000 annotation. The corresponding
    all-voxel archive is nsd_data_unfiltered.pkl.

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
    nsd = utils.load_pickle(os.path.join(root, cfg.get("nsd_data_file", "nsd_data.pkl")))

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
    Load NCSNR-filtered NSD fMRI responses for requested subjects and regions.

    ``nsd_data.pkl`` is the reliable-voxel default for all evaluations. The
    preprocessing pipeline preserves all ROI voxels separately in
    ``nsd_data_unfiltered.pkl``; select it with ``cfg.nsd_data_file``.

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
    nsd = utils.load_pickle(os.path.join(root, cfg.get("nsd_data_file", "nsd_data.pkl")))
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


# ──────────────────── NSD-SYNTHETIC ─────────────────────
class NsdSyntheticTransform:
    """Input transform used by the NSD-synthetic authors (Gifford et al. 2025).

    The raw images are 714x1360: a centred 714x714 content square with grey
    padding either side. Squaring with ``CenterCrop`` before resizing keeps the
    whole square, unlike the standard ``Resize(256) + CenterCrop(224)`` pipeline,
    which clips a 45-pixel band and truncates the word-position stimuli.
    Reference: gifale95/NSD-synthetic, paper_figure_4/01_extract_nsdsynthetic_image_features.py
    """

    def __init__(self, ds_stats: str = "imgnet"):
        self.after_crop = transforms.Compose([
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(DS_MEAN[ds_stats], DS_STD[ds_stats]),
        ])

    def __call__(self, image):
        linear = (np.sqrt(np.asarray(image.convert("RGB")) / 255) * 255).astype(np.uint8)
        square = Image.fromarray(linear)
        return self.after_crop(transforms.CenterCrop(min(square.size))(square))

    def __repr__(self):
        return "NsdSyntheticTransform(sqrt, CenterCrop(min_size), Resize(224), ToTensor, Normalize)"


# Words at retinal positions 1 and 5 lie outside the 714x714 content square, so these
# 16 stimuli are blank after cropping: pixel-identical images that inflate RDM structure.
_BLANK_AFTER_CROP = {f"word{n}_pos{p}_{i}" for n in (4, 6) for p in (1, 5) for i in range(1, 5)}


def load_all_nsd_synthetic_data(cfg: Dict, subjects=None, regions=None) -> Dict:
    """Load NCSNR-filtered NSD-synthetic responses: 204 shared stimuli, test only.

    Same subjects and region names as NSD, so an eval config can swap
    ``neural_dataset`` from ``nsd`` to ``nsd_synthetic`` and nothing else. There
    is no train split — the per-ROI layer comes from the matching regular-NSD run.

    Returns:
        dict with keys "regions", "subjects", "neural" ({region: {subj: {sid: resp}}}),
        "stimuli" ({sid: png path}) and "test_ids".
    """
    subjects = subjects if subjects is not None else _NSD_SUBJECTS
    region_pairs = [(pkl_key, name) for name, pkl_key in _NSD_REGION_MAP.items()
                    if regions is None or name in regions]

    root = os.path.join("datasets", "neural", "nsd_synthetic")
    synth = utils.load_pickle(os.path.join(
        root, cfg.get("nsd_synthetic_data_file", "nsd_synthetic_data.pkl")))
    test_ids = [sid for sid in synth["shared_stimulus_names"] if sid not in _BLANK_AFTER_CROP]

    neural = {
        name: {subj: {sid: synth["data"][pkl_key][subj].sel(stimulus=sid).values
                      for sid in test_ids}
               for subj in subjects}
        for pkl_key, name in region_pairs
    }
    stimuli = {sid: os.path.join(root, "stimuli", f"{sid}.png") for sid in test_ids}

    region_names = [name for _, name in region_pairs]
    logger.info(
        f"Loaded NSD-synthetic: {len(subjects)} subjects x {len(region_names)} regions, "
        f"{len(test_ids)} stimuli (NCSNR > {synth['ncsnr_threshold']})"
    )
    return {"regions": region_names, "subjects": list(subjects),
            "neural": neural, "stimuli": stimuli, "test_ids": test_ids}


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


# ─────────────────── TVSD All Subjects ────────────────────
_TVSD_REGIONS = ["V1", "V4", "IT"]
_TVSD_SUBJECTS = [0, 1]


def load_all_tvsd_data(cfg: Dict, subjects=None, regions=None) -> Dict:
    """
    Load TVSD macaque MUA responses for requested subjects and regions.

    Preprocessing keeps electrodes with mean reliability > 0.3, the threshold used
    by Papale et al. (2025) when preparing TVSD for model training.

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
    mp_kwargs = ({"persistent_workers": True, "prefetch_factor": 2}
                 if workers > 0 else {})
    return DataLoader(
        _StimuliDataset(stimuli, transform),
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        collate_fn=custom_collate_fn,
        pin_memory=torch.cuda.is_available(),
        **mp_kwargs,
    )


def get_neural_loader(cfg: Dict) -> Tuple[Dict[str, Any], DataLoader]:
    """Returns (targets, dataloader) for THINGS-behavior (the only dataset evaluated
    through a single loader; NSD/TVSD go through load_all_*_data)."""
    if cfg.get("neural_dataset") != "things-behavior":
        raise ValueError("get_neural_loader only supports 'things-behavior'")
    targets, stimuli = load_things_data()
    dataloader = _make_loader(stimuli, get_transform(ds_stats="imgnet"), cfg["batchsize"], cfg["num_workers"])
    return targets, dataloader
