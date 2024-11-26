import os
import shutil

# Path to the validation data and annotations
val_dir = 'data/tiny-imagenet-200/val'
val_img_dir = os.path.join(val_dir, 'images')
val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')

# Read the annotations file
val_img_dict = {}
with open(val_annotations_file, 'r') as f:
    for line in f.readlines():
        parts = line.strip().split('\t')
        img_name = parts[0]
        img_label = parts[1]
        val_img_dict[img_name] = img_label

# Create subdirectories and move images
for img_name, img_label in val_img_dict.items():
    label_dir = os.path.join(val_dir, img_label)
    if not os.path.exists(label_dir):
        os.makedirs(label_dir)
    src = os.path.join(val_img_dir, img_name)
    dst = os.path.join(label_dir, img_name)
    shutil.move(src, dst)

# Remove the now-empty 'images' directory
os.rmdir(val_img_dir)
