import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import StandardScaler


def preprocess_images(image_dir, image_size=(64, 64)):
    """
    Preprocess images: Resize, flatten, normalize, and label
    
    Args:
        image_dir (str): Path to dataset directory.
        image_size (tuple): Target size for resizing (width, height).
        
    Returns:
        tuple: Preprocessed data (numpy array), labels (numpy array), and class mapping.
    """
    data, labels = [], []
    class_mapping = {}
    classes = sorted(os.listdir(image_dir))
    
    for class_idx, class_name in enumerate(classes):
        class_mapping[class_idx] = class_name
        class_path = os.path.join(image_dir, class_name)
        
        if os.path.isdir(class_path):
            for image_file in os.listdir(class_path):
                image_path = os.path.join(class_path, image_file)
                try:
                    image = Image.open(image_path).convert('RGB')
                    image_resized = image.resize(image_size)
                    flattened_image = np.asarray(image_resized).flatten()
                    data.append(flattened_image)
                    labels.append(class_idx)
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")

    # normalize data
    data = np.array(data)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    
    return np.array(data), np.array(labels), class_mapping

if __name__ == "__main__":
    train_data, train_labels, class_mapping = preprocess_images("Condensed_Dataset/new_train/")
    print(f"Processed {len(train_data)} training samples with {len(class_mapping)} classes.")
