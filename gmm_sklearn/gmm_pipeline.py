import os
import numpy as np
from sklearn.mixture import GaussianMixture
from data_loader import load_and_preprocess_dataset
from metrics import evaluate_metrics
from visualization import visualize_clusters, save_sample_images

def main(train_dir, test_dir, output_dir, n_components=100, max_iters=10, pca_components=50, cluster_images=False):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load datasets with PCA
    train_data, train_labels, class_names = load_and_preprocess_dataset(train_dir, n_components=pca_components)
    test_data, test_labels, _ = load_and_preprocess_dataset(test_dir, n_components=pca_components)
    
    # Adjust n_components dynamically based on dataset size
    max_components = train_data.shape[0]
    if n_components > max_components:
        print(f"Warning: n_components ({n_components}) exceeds the number of samples ({max_components}). Adjusting to {max_components}.")
        n_components = max_components
    
    # Train GMM
    gmm = GaussianMixture(n_components=n_components, max_iter=max_iters, random_state=42)
    gmm.fit(train_data)
    
    # Predict labels
    predicted_labels = gmm.predict(test_data)
    
    # Evaluate metrics
    evaluate_metrics(
        true_labels=test_labels, 
        predicted_labels=predicted_labels, 
        class_names=class_names, 
        output_dir=output_dir
    )
    
    # Visualize clusters if enabled
    if cluster_images:
        visualize_clusters(test_data, test_labels, predicted_labels, output_dir)
        save_sample_images(
            data_dir=test_dir, 
            cluster_assignments=predicted_labels, 
            labels=test_labels, 
            output_dir=os.path.join(output_dir, "sample_images"),
            image_size=(224, 224)
        )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run GMM pipeline with PCA and clustering visualization')
    parser.add_argument('--train_dir', type=str, required=True, help='Path to training dataset')
    parser.add_argument('--test_dir', type=str, required=True, help='Path to test dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save outputs')
    parser.add_argument('--n_components', type=int, default=100, help='Number of GMM components')
    parser.add_argument('--max_iters', type=int, default=10, help='Maximum number of iterations for GMM')
    parser.add_argument('--pca_components', type=int, default=50, help='Number of PCA components')
    parser.add_argument('--cluster_images', action='store_true', help='Enable clustering and visualization')

    args = parser.parse_args()
    main(args.train_dir, args.test_dir, args.output_dir, args.n_components, args.max_iters, args.pca_components, args.cluster_images)
