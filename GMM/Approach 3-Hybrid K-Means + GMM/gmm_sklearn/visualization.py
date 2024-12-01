import os
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from sklearn.manifold import TSNE

def visualize_clusters(data, labels, cluster_assignments, output_dir, sample_size=100):
    """
    Visualize clusters using t-SNE to reduce dimensionality to 2D for plotting.
    """
    # Ensure sample size does not exceed dataset size
    sample_size = min(sample_size, data.shape[0])
    
    # Dynamically adjust perplexity
    perplexity = min(30, sample_size - 1)  # t-SNE requires perplexity < n_samples
    if perplexity < 5:
        perplexity = 5  # Set a minimum perplexity for meaningful results
    
    print(f"Using t-SNE with perplexity={perplexity} and sample size={sample_size}")
    
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reduced_data = tsne.fit_transform(data[:sample_size])
    reduced_labels = labels[:sample_size]
    reduced_clusters = cluster_assignments[:sample_size]

    plt.figure(figsize=(12, 8))
    for cluster in np.unique(reduced_clusters):
        idx = np.where(reduced_clusters == cluster)
        plt.scatter(reduced_data[idx, 0], reduced_data[idx, 1], label=f'Cluster {cluster}', alpha=0.7)
    
    plt.legend()
    plt.title("Cluster Visualization (t-SNE)")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.savefig(os.path.join(output_dir, "cluster_visualization.png"))
    plt.show()

def save_sample_images(data_dir, cluster_assignments, labels, output_dir, image_size=(224, 224)):
    """
    Save original sample images for each cluster in a folder for visual inspection.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image paths
    image_paths = []
    class_names = sorted(os.listdir(data_dir))  # Ensure consistent class ordering
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        image_paths.extend([os.path.join(class_dir, img) for img in os.listdir(class_dir)])
    
    # Save images to their clusters
    for cluster in np.unique(cluster_assignments):
        cluster_dir = os.path.join(output_dir, f"cluster_{cluster}")
        os.makedirs(cluster_dir, exist_ok=True)
        
        cluster_indices = np.where(cluster_assignments == cluster)[0]
        for idx in cluster_indices[:10]:  # Save up to 10 images per cluster
            img_path = image_paths[idx]  # Get original image path
            img = load_img(img_path, target_size=image_size)  # Load and resize image
            img_name = f"image_{idx}_label_{labels[idx]}.png"
            img_save_path = os.path.join(cluster_dir, img_name)
            img.save(img_save_path)  # Save original image to cluster folder
