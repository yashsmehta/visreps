import os
import functools
from multiprocessing import Pool
from PIL import Image
import torch


def get_train_image_paths(train_dir):
    """Retrieve all image paths from the training directory of Tiny ImageNet."""
    image_paths = []
    for class_dir in os.listdir(train_dir):
        images_path = os.path.join(train_dir, class_dir, "images")

        if os.path.isdir(images_path):
            image_paths.extend(
                os.path.join(images_path, f)
                for f in os.listdir(images_path)
                if f.endswith(".JPEG")
            )

    return image_paths

def process_image(preprocess, image_path):
    """Function to open and preprocess an image."""
    image = Image.open(image_path).convert('RGB')
    return preprocess(image)

def load_and_transform_images(image_paths, preprocess):
    """ Load images from paths and apply the specified torchvision transform. """
    
    partial_process_image = functools.partial(process_image, preprocess)
    
    with Pool() as pool:
        processed_images = pool.map(partial_process_image, image_paths)
    
    return torch.stack(processed_images)
