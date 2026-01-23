#!/usr/bin/env python3
import os
import random
import shutil
import argparse
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

def create_imagenet_mini(imagenet_dir: Path, num_images: int = 10):
    """Creates a mini version of ImageNet by sampling images from each class."""
    dest_dir = imagenet_dir.parent / f"imagenet-mini-{num_images}"
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating ImageNet Mini at: {dest_dir}")
    print(f"Sampling {num_images} images per class")
    
    class_dirs = sorted([d for d in imagenet_dir.iterdir() if d.is_dir()])
    print(f"Found {len(class_dirs)} classes")
    
    for class_dir in tqdm(class_dirs, desc="Processing classes"):
        dest_class_dir = dest_dir / class_dir.name
        dest_class_dir.mkdir(exist_ok=True)
        
        # Get all image files
        image_files = [f for f in class_dir.iterdir() 
                      if f.suffix.lower() in ('.jpg', '.jpeg', '.png')]
        
        # Sample or take all if fewer than requested
        selected = (random.sample(image_files, num_images) 
                   if len(image_files) >= num_images else image_files)
        
        # Copy selected files
        for src_file in selected:
            shutil.copy2(src_file, dest_class_dir / src_file.name)
    
    print(f"Dataset creation complete at {dest_dir}")

if __name__ == "__main__":
    load_dotenv()
    
    imagenet_dir = os.environ.get("IMAGENET_DATA_DIR")
    if not imagenet_dir:
        print("Error: IMAGENET_DATA_DIR not set in .env file")
        exit(1)
    
    parser = argparse.ArgumentParser(description="Create mini ImageNet dataset")
    parser.add_argument("--num_images", type=int, default=50,
                       help="Number of images per class")
    args = parser.parse_args()
    
    create_imagenet_mini(Path(imagenet_dir).expanduser(), args.num_images)