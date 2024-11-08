"""Data Cleaning - ML Car Model"""

import os
import shutil
from PIL import Image, ImageOps, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Dataset paths
train_dir = "/content/train"
test_dir = "/content/test"
target_size = (224, 224)

# Function to delete folders with <= 30 images
def delete_small_folders(data_dir):
    folder_counts = {}
    res = 0

    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)

        if os.path.isdir(folder_path):
            image_count = len(os.listdir(folder_path))
            if image_count <= 30:
                res += 1
                folder_counts[folder_name] = image_count
                shutil.rmtree(folder_path)
                print(f"Deleted folder '{folder_name}' with {image_count} images.")

    print(f"Deleted {res} folders with 30 or fewer images.")

# Resize images to consistent size with padding
def resize_images(data_dir, target_size):
    resized_images = {}
    
    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)

        if os.path.isdir(folder_path):
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)

                try:
                    with Image.open(image_path) as img:
                        if img.size != target_size:
                            img.thumbnail(target_size, Image.LANCZOS)
                            img_padded = ImageOps.pad(img, target_size, color=(255, 255, 255))
                            img_padded.save(image_path)
                            resized_images[image_name] = img.size
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
    
    print("Resized images summary:")
    for img, size in resized_images.items():
        print(f"{img} was resized from {size} to {target_size}")

# Check for non-JPEG images
def check_image_format(data_dir, extension=".jpg"):
    for folder_name in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder_name)

        if os.path.isdir(folder_path):
            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)

                if not image_name.lower().endswith(extension):
                    print(f"Non-JPEG image found: {image_path}")

# Run cleaning steps on train and test datasets
delete_small_folders(train_dir)
resize_images(train_dir, target_size)
check_image_format(train_dir)

delete_small_folders(test_dir)
resize_images(test_dir, target_size)
check_image_format(test_dir)
