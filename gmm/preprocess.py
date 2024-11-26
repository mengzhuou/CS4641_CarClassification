import os
import cv2
import numpy as np


def preprocess_images(image_dir, image_size=(224, 224)):
    """
    Preprocess images into numpy arrays with normalization.

    Args:
        image_dir (str): Path to the dataset directory.
        image_size (tuple): Size to resize images (height, width).

    Returns:
        tuple: (data, labels, label_map) where data is the image data,
               labels are the corresponding labels, and label_map maps label indices to class names.
    """
    data = []
    labels = []
    label_map = {}

    for idx, label in enumerate(sorted(os.listdir(image_dir))):
        label_map[idx] = label
        label_dir = os.path.join(image_dir, label)
        for file in os.listdir(label_dir):
            file_path = os.path.join(label_dir, file)
            image = cv2.imread(file_path)
            image = cv2.resize(image, image_size)
            image = image / 255.0  # Normalize pixel values
            data.append(image)
            labels.append(idx)

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    return data, labels, label_map

if __name__ == "__main__":
    train_data, train_labels, class_mapping = preprocess_images("Condensed_Dataset/new_train/")
    print(f"Processed {len(train_data)} training samples with {len(class_mapping)} classes.")
