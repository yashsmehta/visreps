#!/usr/bin/env python3
import os
import random
import shutil
import argparse
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_imagenet_mini(imagenet_dir_str: str, num_images_per_class: int = 50):
    """
    Creates a mini version of the ImageNet training set.

    Args:
        imagenet_dir_str: Path to the root ImageNet directory (containing 'train', 'val').
        num_images_per_class: Number of images to sample from each class.
    """
    load_dotenv() # Load environment variables from .env file

    if not imagenet_dir_str:
        logging.error("IMAGENET_DATA_DIR environment variable not set or provided.")
        return

    imagenet_dir = Path(imagenet_dir_str).expanduser()
    # Assume class directories are directly under imagenet_dir
    source_dir = imagenet_dir 

    if not source_dir.is_dir():
        logging.error(f"Source ImageNet directory not found: {source_dir}")
        return

    # Define destination path adjacent to the original ImageNet directory
    mini_dataset_name = f"imagenet-mini-{num_images_per_class}"
    dest_dir = imagenet_dir.parent / mini_dataset_name

    logging.info(f"Creating ImageNet Mini dataset at: {dest_dir}")
    logging.info(f"Sampling {num_images_per_class} images per class.")

    try:
        # Create base destination directory
        dest_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logging.error(f"Error creating destination directory {dest_dir}: {e}")
        return

    class_dirs = sorted([d for d in source_dir.iterdir() if d.is_dir()])
    logging.info(f"Found {len(class_dirs)} classes in {source_dir}.")

    if not class_dirs:
        logging.error(f"No class subdirectories found in {source_dir}.")
        return

    for class_dir in tqdm(class_dirs, desc="Processing classes"):
        class_name = class_dir.name
        dest_class_dir = dest_dir / class_name

        try:
            dest_class_dir.mkdir(exist_ok=True)
        except OSError as e:
            logging.warning(f"Could not create destination class directory {dest_class_dir}: {e}. Skipping class {class_name}.")
            continue

        try:
            image_files = sorted([f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        except OSError as e:
            logging.warning(f"Could not read files from source class directory {class_dir}: {e}. Skipping class {class_name}.")
            continue

        if len(image_files) < num_images_per_class:
            logging.warning(f"Class {class_name} has only {len(image_files)} images, less than the required {num_images_per_class}. Copying all available images.")
            selected_files = image_files
        else:
            selected_files = random.sample(image_files, num_images_per_class)

        for src_file_path in selected_files:
            dest_file_path = dest_class_dir / src_file_path.name
            try:
                shutil.copy2(src_file_path, dest_file_path) # copy2 preserves metadata
            except Exception as e:
                logging.warning(f"Could not copy {src_file_path} to {dest_file_path}: {e}")

    logging.info(f"ImageNet Mini dataset creation complete at {dest_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a mini ImageNet dataset by sampling images from the original.")
    parser.add_argument(
        "--imagenet_dir",
        type=str,
        default=os.environ.get("IMAGENET_DATA_DIR"),
        help="Path to the root ImageNet directory. Defaults to IMAGENET_DATA_DIR environment variable.",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=50,
        help="Number of images to sample per class.",
    )
    args = parser.parse_args()

    create_imagenet_mini(args.imagenet_dir, args.num_images)