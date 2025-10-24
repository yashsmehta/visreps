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
from bonner.datasets.hebart2022_things_behavior._data import (
    load_embeddings as _load_things_embed,
)
from bonner.datasets.hebart2019_things._stimuli import StimulusSet as _ThingsStimSet

logger = logging.getLogger(__name__)


# ───────────────────────── NSD ──────────────────────────
def load_nsd_data(cfg: Dict) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Load fMRI responses and stimulus images for a given NSD subject and brain region.

    Args:
        cfg (Dict): Contains "region" and "subject_idx".

    Returns:
        (targets, stimuli): Dicts mapping stimulus IDs to response arrays and image arrays.
    """
    region, subj = cfg["region"], cfg["subject_idx"]
    root = utils.get_env_var("NSD_DATA_DIR")
    fmri_xr = utils.load_pickle(os.path.join(root, "fmri_responses.pkl"))[region][subj]
    
    stimulus_ids = [int(i) for i in fmri_xr.coords["stimulus"].values]
    
    # Load images directly from HDF5 file
    hdf5_path = "/data/shared/datasets/allen2021.natural_scenes/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"
    images = {}
    
    with h5py.File(hdf5_path, "r") as f:
        imgBrick = f["imgBrick"]
        sorted_indices = np.sort(stimulus_ids)
        loaded_images = imgBrick[sorted_indices]
        
        for i, stim_id in enumerate(sorted_indices):
            images[str(stim_id)] = loaded_images[i]
    
    ids = {str(i) for i in stimulus_ids}
    
    return (
        {i: fmri_xr.sel(stimulus=int(i)).values for i in ids},
        {i: images[i] for i in ids},
    )


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
def load_things_data() -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """
    Load THINGS behavioral embeddings and corresponding image file paths.

    Returns:
        embeds (dict[str, np.ndarray]): Mapping from object ID to embedding vector (float32).
        img_paths (dict[str, str]): Mapping from object ID to absolute image file path.
            Only includes images with valid, existing paths as resolved via _ThingsStimSet logic.
    """
    beh_xr = _load_things_embed()
    embeds = {
        str(i): beh_xr.sel(object=i).values.astype(np.float32)
        for i in beh_xr["object"].values
    }

    img_paths = {}
    stimset = _ThingsStimSet()
    for k in stimset.metadata.index:
        try:
            relative_filename = stimset.metadata.loc[k, "filename"]
            relative_path_part = relative_filename.split("images/", 1)[-1]
            correct_path = os.path.join(
                stimset.root, "images", "object_images", relative_path_part
            )
            if os.path.exists(correct_path):
                img_paths[k] = correct_path
            else:
                logger.warning(
                    f"Constructed path {correct_path} for image {k} does not exist. Skipping."
                )
        except KeyError:
            logger.error(f"Key {k} not found in _ThingsStimSet metadata. Skipping.")
        except Exception as e:
            logger.warning(
                f"Error constructing path for image {k}: {e}. Skipping.", exc_info=False
            )

    return embeds, img_paths


# ─────────────────────── Dataset/Loader ───────────────────────
class _StimuliDataset(Dataset):
    """
    PyTorch Dataset for stimuli, supporting both file paths and in-memory image data.

    Args:
        stimuli (Dict[str, Any]): Mapping from stimulus ID to file path, np.ndarray, or PIL.Image.
        transform (callable): Transform to apply to each image.
    """

    def __init__(self, stimuli: Dict[str, Any], transform):
        self.keys = sorted(stimuli.keys())
        self.stimuli = {k: stimuli[k] for k in self.keys}
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
    Supported datasets: 'nsd', 'things', 'nsd_synthetic', 'cusack'.
    """
    dataset_type = cfg.get("neural_dataset")
    if dataset_type == "nsd":
        targets, stimuli = load_nsd_data(cfg)
    elif dataset_type == "things":
        targets, stimuli = load_things_data()
    elif dataset_type == "nsd_synthetic":
        targets, stimuli = load_nsd_synthetic_data(cfg)
    elif dataset_type == "cusack":
        targets, stimuli = load_cusack_data(cfg)
    else:
        raise ValueError("neural_dataset must be 'nsd', 'things', 'nsd_synthetic', or 'cusack'")

    transform = get_transform(ds_stats="imgnet")
    dataloader = _make_loader(
        stimuli,
        transform,
        cfg["batchsize"],
        cfg["num_workers"],
    )
    return targets, dataloader
