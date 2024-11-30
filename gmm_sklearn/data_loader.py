import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_and_preprocess_dataset(directory, img_height=224, img_width=224, n_components=None):
    data, labels = [], []
    class_names = os.listdir(directory)
    encoder = LabelEncoder()
    encoder.fit(class_names)
    
    for label in class_names:
        folder_path = os.path.join(directory, label)
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            img = load_img(img_path, target_size=(img_height, img_width))
            img_array = img_to_array(img) / 255.0  # Normalize images
            data.append(img_array.flatten())
            labels.append(label)
    
    encoded_labels = encoder.transform(labels)
    data = np.array(data)
    
    # Determine the max allowable components for PCA
    max_components = min(data.shape[0], data.shape[1])
    if n_components:
        if n_components > max_components:
            print(f"Warning: n_components ({n_components}) exceeds the allowable limit ({max_components}). Adjusting to {max_components}.")
            n_components = max_components
        
        # Apply PCA
        pca = PCA(n_components=n_components, random_state=42)
        data = pca.fit_transform(data)
        print(f"PCA reduced data to {n_components} dimensions")
    
    return data, np.array(encoded_labels), class_names
