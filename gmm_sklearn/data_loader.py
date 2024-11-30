import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Initialize the pre-trained CNN model
CNN_MODEL = ResNet50(weights='imagenet', include_top=False, pooling='avg')

def extract_deep_features(image_paths, model, input_size=(224, 224), batch_size=32):
    """
    Extract deep features from images using a pre-trained CNN model with batching.
    """
    features = []
    num_images = len(image_paths)

    for i in range(0, num_images, batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []

        for img_path in batch_paths:
            img = load_img(img_path, target_size=input_size)
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)
            batch_images.append(img_array)

        batch_images = np.array(batch_images)
        batch_features = model.predict(batch_images, verbose=0)
        features.extend(batch_features)

    return np.array(features)

def load_and_preprocess_dataset(directory, img_height=224, img_width=224, n_components=None, use_cnn=True):
    """
    Load and preprocess the dataset with optional PCA and CNN feature extraction.
    """
    data, labels = [], []
    class_names = os.listdir(directory)
    encoder = LabelEncoder()
    encoder.fit(class_names)
    
    image_paths = []
    
    for label in class_names:
        folder_path = os.path.join(directory, label)
        for img_file in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_file)
            image_paths.append(img_path)
            labels.append(label)
    
    encoded_labels = encoder.transform(labels)
    
    if use_cnn:
        print("Extracting deep features using a pre-trained CNN model...")
        data = extract_deep_features(image_paths, CNN_MODEL, input_size=(img_height, img_width))
    else:
        print("Loading raw image data...")
        for img_path in image_paths:
            img = load_img(img_path, target_size=(img_height, img_width))
            img_array = img_to_array(img) / 255.0  # Normalize images
            data.append(img_array.flatten())
        data = np.array(data)
        # Normalize the data
        data /= np.linalg.norm(data, axis=1, keepdims=True)
    
    # Determine the max allowable components for PCA
    max_components = min(data.shape[0], data.shape[1])
    if n_components:
        if n_components > max_components:
            print(f"Warning: n_components ({n_components}) exceeds the allowable limit ({max_components}). Adjusting to {max_components}.")
            n_components = max_components
        
        # Apply PCA
        print(f"Applying PCA to reduce features to {n_components} dimensions...")
        pca = PCA(n_components=n_components, random_state=42)
        data = pca.fit_transform(data)
        print(f"PCA reduced data to {n_components} dimensions")
    
    return data, np.array(encoded_labels), class_names
