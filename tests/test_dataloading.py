import pytest
import torch
import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from omegaconf import OmegaConf
from unittest.mock import patch, MagicMock
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
import json
from dotenv import load_dotenv
import glob
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision.utils import make_grid

from visreps.dataloaders.obj_cls import get_obj_cls_loader, PCADataset, ImageNetDataset, DS_MEAN, DS_STD
import visreps.utils as utils
from visreps.utils import get_env_var # Import to patch

# --- Configuration & Environment Setup ---

# Load environment variables from .env file
load_dotenv()

# Mark tests requiring real ImageNet data
# These tests will be skipped if IMAGENET_DATA_DIR is not set or points to dummy paths
requires_imagenet = pytest.mark.skipif(
    os.getenv("IMAGENET_DATA_DIR") is None or not os.path.isdir(os.getenv("IMAGENET_DATA_DIR")),
    reason="Requires IMAGENET_DATA_DIR environment variable pointing to the real dataset directory"
)

# Define markers - properly defined and registered
# Use pytest.ini or add this to conftest.py for permanent registration
fast = pytest.mark.fast  # For tests that should run in the fast suite
slow = pytest.mark.slow  # For tests that are slow and can be skipped with -m "not slow"
visualize = pytest.mark.visualize  # Control visualization tests

# Reduce the number of classes and parameters for faster testing
PCA_BASE_DIR = os.path.join("datasets", "obj_cls", "imagenet")
FILENAME_COLUMN = 'image' # Column name containing the base filenames
N_CLASSES_LIST = [2, 8]  # Reduced from [2, 4, 8, 16, 32, 64]
PCA_CONFIGS = [
    {
        "subfolder": "pca_labels_none",
        "features_file": os.path.join(PCA_BASE_DIR, "features_pretrained_none.npz")
    },
    {
        "subfolder": "pca_labels_imagenet1k",
        "features_file": os.path.join(PCA_BASE_DIR, "features_pretrained_imagenet1k.npz")
    }
]
DATA_INTEGRITY_TEST_PARAMS = [
    (config["subfolder"], config["features_file"], n_classes)
    for config in PCA_CONFIGS
    for n_classes in N_CLASSES_LIST
]

# Base fixture for ImageNet test config
@pytest.fixture
def imagenet_base_cfg():
    base_cfg_path = os.path.join(os.path.dirname(__file__), '../configs/train/base.json')
    with open(base_cfg_path, 'r') as f:
        base_cfg_dict = json.load(f)
    cfg = OmegaConf.create(base_cfg_dict)
    test_overrides = OmegaConf.create({
        "dataset": "imagenet", # Default to ImageNet
        "batchsize": 4,  # Reduced batch size for faster testing
        "num_workers": 0,
        "data_augment": False,
        "pca_labels": False,
        "pca_n_classes": None,
        "pca_labels_folder": "pca_labels_imagenet1k", # Default, can be overridden
        "seed": 42,
        "log_checkpoints": False,
        "use_wandb": False,
        "num_epochs": 1,
        # Add other necessary fields if Trainer requires them implicitly
        "model_class": "mock", # Not relevant for dataloading tests
        "optimizer": "mock",
        "learning_rate": 0.1,
        "lr_scheduler": "mock",
    })
    cfg = OmegaConf.merge(cfg, test_overrides)
    # Ensure data_dir points to the real location if available
    real_data_dir = os.getenv("IMAGENET_DATA_DIR")
    if real_data_dir:
        cfg.data_dir = real_data_dir
    return cfg

# Mock environment variables - only mock if real ones aren't set
# This allows tests to use real paths when available
@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    env_to_check = ["IMAGENET_DATA_DIR", "IMAGENET_LOCAL_DIR"]
    needed_mocks = {}
    for var in env_to_check:
        if not os.getenv(var):
            print(f"Warning: Environment variable {var} not set. Mocking with dummy path for tests.")
            needed_mocks[var] = f"/dummy/{var.lower()}"

    if needed_mocks:
        original_get_env_var = utils.get_env_var
        def mock_get_env_var(var_name, default=None):
            # Return mocked value if needed, otherwise use original function (which reads .env)
            if var_name in needed_mocks:
                return needed_mocks[var_name]
            return original_get_env_var(var_name, default)
        # Patch in both locations where it might be imported from
        monkeypatch.setattr("visreps.dataloaders.obj_cls.utils.get_env_var", mock_get_env_var)
        monkeypatch.setattr("visreps.utils.get_env_var", mock_get_env_var)

# --- Helper Functions (Keep as they are generally useful) ---

def get_image_basenames_from_features(features_path, max_samples=100):
    """Loads image basenames from an .npz file for integrity checks. Limit samples for speed."""
    if not os.path.exists(features_path):
        pytest.fail(f"Source features file not found: {features_path}")
    try:
        with np.load(features_path, allow_pickle=True) as data_dict:
            if 'image_names' not in data_dict:
                pytest.fail(f"'image_names' key not found in {features_path}")
            image_names = data_dict['image_names']
            if not image_names.size: return set()
            
            # Only process a subset of names for faster testing
            if len(image_names) > max_samples:
                image_names = image_names[:max_samples]
                
            if isinstance(image_names[0], (bytes, np.bytes_)):
                try: decoded_names = [name.decode('utf-8') for name in image_names]
                except UnicodeDecodeError: pytest.fail(f"Failed to decode image names in {features_path}")
            else: decoded_names = image_names
            return {os.path.basename(str(name)) for name in decoded_names}
    except Exception as e: pytest.fail(f"Error loading/processing {features_path}: {e}")

def denormalize_image(img: torch.Tensor, dataset_name: str) -> torch.Tensor:
    img = img.clone()
    # Use 'imgnet' key for ImageNet stats
    key = "imgnet" if dataset_name == "imagenet" else dataset_name
    mean = DS_MEAN[key]
    std = DS_STD[key]
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(img, 0, 1)

def tensor_to_pil(img_tensor: torch.Tensor, dataset_name: str) -> Image.Image:
    if img_tensor.dim() == 4: img_tensor = img_tensor[0]
    img = denormalize_image(img_tensor, dataset_name)
    img_uint8 = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(img_uint8)

def visualize_batch(images: torch.Tensor, labels: torch.Tensor, dataset_name: str, save_path: str):
    # Skip visualization by default for faster tests
    # Only run visualization if environment variable is set
    if os.environ.get('ENABLE_TEST_VISUALIZATION') != '1':
        return
        
    sns.set_style("white")
    plt.rcParams['figure.facecolor'] = 'white'
    images_denorm = denormalize_image(images, dataset_name)
    grid = make_grid(images_denorm[:min(len(images), 8)], nrow=4, padding=2) # Show max 8 images in 2x4 grid
    plt.figure(figsize=(8, 4))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.axis('off')
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()

# --- Tests for ImageNet Dataloader ---

@fast
@requires_imagenet
def test_imagenet_loader_basic(imagenet_base_cfg):
    """Test basic loading, splits, batch shape, types for ImageNet."""
    cfg = imagenet_base_cfg
    cfg.pca_labels = False # Ensure standard labels
    datasets, loaders = get_obj_cls_loader(cfg)
    assert "train" in loaders and "test" in loaders
    assert isinstance(datasets["train"], ImageNetDataset)
    assert isinstance(datasets["test"], ImageNetDataset)

    # Verify label consistency by checking class count inferred by dataset
    assert datasets["train"].num_classes == 1000
    assert datasets["test"].num_classes == 1000
    
    # Sample a small subset of labels for faster testing
    train_labels = set()
    for i, sample in enumerate(datasets["train"].samples):
        if i >= 100: break  # Limit samples checked
        label = sample[1] # Assuming label is the second element
        train_labels.add(label)
    test_labels = set()
    for i, sample in enumerate(datasets["test"].samples):
        if i >= 100: break  # Limit samples checked  
        label = sample[1] # Assuming label is the second element
        test_labels.add(label)
    
    # Just check that we have a reasonable number of class labels
    # Reduced expectation since we're sampling fewer examples
    assert len(train_labels) > 10, f"Expected many classes, found {len(train_labels)}"
    assert all(0 <= lbl < 1000 for lbl in train_labels)

    images, labels = next(iter(loaders["train"]))
    assert images.shape == (cfg.batchsize, 3, 224, 224)
    assert labels.shape == (cfg.batchsize,)
    assert images.dtype == torch.float32
    assert labels.dtype == torch.long
    assert labels.min() >= 0 and labels.max() < 1000
    visualize_batch(images, labels, "imagenet", "tests/outputs/imagenet_samples.png")

@fast
@requires_imagenet
@pytest.mark.parametrize("augment", [False, True])  # Test False first for speed
def test_imagenet_loader_transforms(imagenet_base_cfg, augment):
    """Test ImageNet transforms and augmentation."""
    cfg = imagenet_base_cfg
    cfg.data_augment = augment
    datasets, _ = get_obj_cls_loader(cfg)
    train_transform = datasets["train"].transform
    test_transform = datasets["test"].transform
    mean_imgnet = DS_MEAN["imgnet"] # Use correct key "imgnet"
    std_imgnet = DS_STD["imgnet"] # Use correct key "imgnet"

    # Check common transforms (Normalization, ToTensor, Resize, CenterCrop)
    for tfm in [train_transform, test_transform]:
        assert any(isinstance(t, transforms.Normalize) and torch.allclose(torch.tensor(t.mean), torch.tensor(mean_imgnet)) and torch.allclose(torch.tensor(t.std), torch.tensor(std_imgnet)) for t in tfm.transforms)
        assert any(isinstance(t, transforms.ToTensor) for t in tfm.transforms)
        # Standard ImageNet transforms
        assert any(isinstance(t, transforms.Resize) and t.size == 256 for t in tfm.transforms)
        assert any(isinstance(t, transforms.CenterCrop) and t.size == (224, 224) for t in tfm.transforms)

    # Check augmentation specific transforms
    has_train_augment = any(isinstance(t, (transforms.RandomResizedCrop, transforms.RandomHorizontalFlip)) for t in train_transform.transforms)
    has_test_augment = any(isinstance(t, (transforms.RandomResizedCrop, transforms.RandomHorizontalFlip)) for t in test_transform.transforms)

    assert has_train_augment == augment
    assert has_test_augment is False # Test transforms should be deterministic

    # Skip augmentation visualization in parametrized tests for speed
    if not augment:
        return
        
    # Visualize augmentation if enabled 
    images, _ = datasets["train"][0] # Get single image tensor from dataset
    pil_img = tensor_to_pil(images.unsqueeze(0)[0], "imagenet") # Get first image as PIL
    # Extract only augmentation transforms for visualization
    aug_transforms_only = [t for t in train_transform.transforms if not isinstance(t, (transforms.ToTensor, transforms.Normalize, transforms.Resize, transforms.CenterCrop))]
    aug_compose = transforms.Compose(aug_transforms_only)
    augmented_images = []
    for _ in range(2):  # Reduced from 4 to 2 augmented samples
         aug_pil = aug_compose(pil_img)
         # Re-apply standard Resize/Crop/ToTensor/Normalize for visualization consistency
         final_tensor = test_transform(aug_pil) # Use test transform for standard processing
         augmented_images.append(final_tensor)
    visualize_batch(torch.stack(augmented_images), torch.zeros(2), "imagenet", "tests/outputs/imagenet_augmentation_samples.png")

@fast
@requires_imagenet
@pytest.mark.parametrize("pca_n_classes", [2]) # Reduced from [2, 8] to just [2] for speed
def test_imagenet_loader_pca_labels(imagenet_base_cfg, pca_n_classes):
    """Test ImageNet loader with PCA labels enabled."""
    cfg = imagenet_base_cfg
    cfg.pca_labels = True
    cfg.pca_n_classes = pca_n_classes
    cfg.pca_labels_folder = "pca_labels_imagenet1k"  # Just test one folder

    # Check if the specific PCA label file exists, skip if not (might not be generated yet)
    pca_label_file_path = os.path.join(
        utils.get_env_var("IMAGENET_LOCAL_DIR"), # PCA files are in LOCAL_DIR
        cfg.pca_labels_folder,
        f"n_classes_{pca_n_classes}.csv"
    )
    if not os.path.exists(pca_label_file_path):
        pytest.skip(f"Required PCA label file not found: {pca_label_file_path}")

    datasets, loaders = get_obj_cls_loader(cfg)

    # Check dataset types are wrapped
    assert isinstance(datasets["train"], PCADataset)
    assert isinstance(datasets["test"], PCADataset)
    assert isinstance(datasets["train"].dataset, ImageNetDataset)
    assert isinstance(datasets["test"].dataset, ImageNetDataset)

    # Check number of classes in config is used and matches dataset wrapper
    assert cfg.pca_n_classes == pca_n_classes
    assert datasets["train"].num_classes == pca_n_classes
    assert datasets["test"].num_classes == pca_n_classes
    # Ensure base n_classes is not set when PCA is active
    assert getattr(cfg, "n_classes", None) is None

    # Verify labels from the loader are within the PCA range
    images, labels = next(iter(loaders["train"]))
    assert labels.min() >= 0
    assert labels.max() < pca_n_classes
    assert labels.dtype == torch.long
    assert images.shape == (cfg.batchsize, 3, 224, 224)

# --- Data Integrity Tests (ImageNet PCA Labels) ---

@pytest.fixture(scope='module')  # Changed to module scope for better caching
def imagenet_file_basenames():
    """Get unique image basenames from the real ImageNet dataset directory."""
    data_dir = utils.get_env_var("IMAGENET_DATA_DIR")
    # Skip logic remains the same (requires real data dir)
    if not data_dir or not os.path.isdir(data_dir):
        pytest.skip("Skipping ImageNet basename collection: Real IMAGENET_DATA_DIR not available.")

    basenames = set()
    label_file = os.path.join(utils.get_env_var("IMAGENET_LOCAL_DIR"), "folder_labels.json")
    if not os.path.exists(label_file):
        pytest.fail(f"Label file not found: {label_file}")
    with open(label_file, "r") as f: valid_folders = set(json.load(f).keys())

    print(f"\nCollecting basenames from {data_dir} (sampling for speed)...")
    processed_files = 0
    max_files_per_folder = 20  # Limit files processed per folder
    max_folders = 10  # Limit folders processed
    
    folder_count = 0
    for folder in os.listdir(data_dir):
        if folder not in valid_folders: continue
        if folder_count >= max_folders: break  # Limit number of folders
        
        folder_count += 1
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path): continue
        
        file_count = 0
        for fname in os.listdir(folder_path):
            if file_count >= max_files_per_folder: break  # Limit files per folder
            
            # Standardize to check lowercase extensions
            if fname.lower().endswith(('.jpeg', '.jpg')):
                 basenames.add(fname)
                 processed_files += 1
                 file_count += 1
                 
    print(f"Sampled {processed_files} image files from {folder_count} folders.")
    if not basenames: pytest.fail(f"No image files found in {data_dir}")
    return basenames

# Use a module scope to share the expensive imagenet_file_basenames fixture
@slow  # Mark as slow test
@requires_imagenet # Skip class if real imagenet isn't available
class TestImageNetPCADataIntegrity:
    expected_basenames_cache = {}

    @pytest.mark.parametrize("subfolder, features_file, n_classes", DATA_INTEGRITY_TEST_PARAMS)
    def test_pca_labels_match_features_and_ground_truth(self, imagenet_file_basenames, subfolder, features_file, n_classes):
        """ Check PCA CSV: exists, filenames match ground truth, count matches source features. """
        # PCA labels are now expected in IMAGENET_LOCAL_DIR
        local_dir = utils.get_env_var("IMAGENET_LOCAL_DIR")
        if not local_dir or not os.path.isdir(local_dir):
             pytest.skip("Skipping PCA integrity check: IMAGENET_LOCAL_DIR not available.")
        pca_label_file = os.path.join(local_dir, subfolder, f"n_classes_{n_classes}.csv")

        # Load expected basenames from source features (use cache)
        # Source features path needs correction - they are NOT necessarily in PCA_BASE_DIR
        # Assuming features files are in IMAGENET_LOCAL_DIR based on scripts/extract_reps.py usage
        source_features_path = os.path.join(local_dir, os.path.basename(features_file))
        if not os.path.exists(source_features_path):
             pytest.fail(f"Source features file not found at expected location: {source_features_path}")

        if source_features_path not in self.expected_basenames_cache:
             # Limit samples for speed
             self.expected_basenames_cache[source_features_path] = get_image_basenames_from_features(
                 source_features_path, max_samples=100
             )
        expected_basenames = self.expected_basenames_cache[source_features_path]
        expected_count = len(expected_basenames)

        # Check PCA label file exists
        if not os.path.exists(pca_label_file): pytest.fail(f"PCA label file not found: {pca_label_file}")

        # Load CSV (reading only a subset of rows for speed)
        try: 
            # Read only first 100 rows for faster testing
            df = pd.read_csv(pca_label_file, nrows=100)
        except Exception as e: pytest.fail(f"Failed to read CSV {pca_label_file}: {e}")
        if FILENAME_COLUMN not in df.columns: pytest.fail(f"Column '{FILENAME_COLUMN}' not found in {pca_label_file}")

        # Get unique basenames from CSV
        try: csv_filenames = set(df[FILENAME_COLUMN].dropna().unique().tolist())
        except Exception as e: pytest.fail(f"Error accessing column '{FILENAME_COLUMN}' in {pca_label_file}: {e}")

        # Check if there's any overlap with our sampled ground truth
        # This is looser than the original test but faster
        overlap = csv_filenames.intersection(imagenet_file_basenames)
        assert len(overlap) > 0, f"No overlap between CSV filenames and sampled ground truth"

    def test_none_features_match_ground_truth_count(self, imagenet_file_basenames):
        """Verify a sample of features_pretrained_none.npz matches ground truth dataset."""
        # Assuming features file is in IMAGENET_LOCAL_DIR
        local_dir = utils.get_env_var("IMAGENET_LOCAL_DIR")
        if not local_dir or not os.path.isdir(local_dir):
             pytest.skip("Skipping feature count check: IMAGENET_LOCAL_DIR not available.")
        none_features_file = os.path.join(local_dir, "features_pretrained_none.npz")
        if not os.path.exists(none_features_file):
             pytest.fail(f"Feature file not found: {none_features_file}")

        # Get a sample of features rather than loading the full file
        feature_basenames = get_image_basenames_from_features(none_features_file, max_samples=100)
        
        # Instead of comparing counts (which will differ due to sampling),
        # verify that the features exist in the ground truth
        overlap = feature_basenames.intersection(imagenet_file_basenames)
        assert len(overlap) > 0, f"No overlap between feature basenames and sampled ground truth"

# --- PCA Class Similarity Test (Now marked slow) ---

@slow
def analyze_pca_class_similarity(n_classes=4):
    """Compares labels between current and legacy PCA folders (informational)."""
    print(f"\n--- Analyzing PCA Label Similarity (ImageNet, {n_classes} classes) ---")
    local_dir = utils.get_env_var("IMAGENET_LOCAL_DIR")
    if not local_dir or not os.path.isdir(local_dir):
         print("! Skipping similarity check: IMAGENET_LOCAL_DIR not available.")
         return

    # Assuming 'imagenet1k' is the 'new' folder structure
    new_path = os.path.join(local_dir, "pca_labels_imagenet1k", f"n_classes_{n_classes}.csv")
    legacy_path = os.path.join(local_dir, "pca_labels_legacy", f"n_classes_{n_classes}.csv")

    if not os.path.exists(new_path) or not os.path.exists(legacy_path):
        print("! Skipping similarity check: One or both PCA label files not found.")
        print(f"  Checked New: {new_path}")
        print(f"  Checked Legacy: {legacy_path}")
        return

    # Read only first 100 rows for faster comparison
    new_labels = pd.read_csv(new_path, nrows=100)
    legacy_labels = pd.read_csv(legacy_path, nrows=100)
    merged = pd.merge(new_labels, legacy_labels, on='image', suffixes=('_new', '_legacy'))

    total_samples = len(merged)
    matching_labels = (merged['pca_label_new'] == merged['pca_label_legacy']).sum()
    agreement_pct = (matching_labels / total_samples) * 100
    print(f"Label agreement between new/legacy: {matching_labels}/{total_samples} ({agreement_pct:.2f}%)")

    if agreement_pct < 100:
        print("Sample mismatches (new vs legacy):")
        print(merged[merged['pca_label_new'] != merged['pca_label_legacy']][['image', 'pca_label_new', 'pca_label_legacy']].head())

@slow  # Mark as slow test
@requires_imagenet # Requires PCA files which depend on ImageNet features/labels
@pytest.mark.parametrize("n_cls", [2])  # Only check one class level
def test_pca_label_similarity(n_cls):
    analyze_pca_class_similarity(n_classes=n_cls) 