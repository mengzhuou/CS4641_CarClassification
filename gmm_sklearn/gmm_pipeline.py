import os
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from data_loader import load_and_preprocess_dataset
from metrics import evaluate_metrics
from visualization import visualize_clusters, save_sample_images

def run_gmm(data, n_components, max_iters, covariance_type):
    """
    Train and predict using Gaussian Mixture Model with dynamic covariance type.
    """
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        max_iter=max_iters,
        random_state=42
    )

    gmm.fit(data)
    predicted_labels = gmm.predict(data)
    return predicted_labels

def run_kmeans(data, n_clusters):
    """
    Train and predict using K-Means clustering.
    """
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    kmeans.fit(data)
    predicted_labels = kmeans.predict(data)
    return predicted_labels

def calculate_silhouette_score(data, labels, method_name, k):
    """
    Calculate and print the silhouette score for clustering.
    """
    silhouette_avg = silhouette_score(data, labels)
    print(f"Silhouette Score for {method_name} with k={k}: {silhouette_avg:.4f}")
    return silhouette_avg

def main(train_dir, test_dir, output_dir, k_values, covariance_types, max_iters=10, pca_components=50, cluster_images=False):
    os.makedirs(output_dir, exist_ok=True)
    
    # Load datasets with with CNN features and PCA
    train_data, train_labels, class_names = load_and_preprocess_dataset(train_dir, n_components=pca_components, use_cnn=True)
    test_data, test_labels, _ = load_and_preprocess_dataset(test_dir, n_components=pca_components, use_cnn=True)
    
    # Iterate over k_values and covariance_types for GMM
    for covariance_type in covariance_types:
        for k in k_values:
            print(f"\nRunning clustering with k={k} and covariance_type={covariance_type}...")

            # Adjust k dynamically based on dataset size
            max_components = train_data.shape[0]
            if k > max_components:
                print(f"Warning: k ({k}) exceeds the number of samples ({max_components}). Adjusting to {max_components}.")
                k = max_components

            # GMM Evaluation
            print(f"  Running GMM with k={k} and covariance_type={covariance_type}...")
            gmm_labels = run_gmm(test_data, n_components=k, max_iters=max_iters, covariance_type=covariance_type)

            gmm_output_dir = os.path.join(output_dir, f"gmm_k{k}_{covariance_type}")
            os.makedirs(gmm_output_dir, exist_ok=True)
        
            # Evaluate metrics
            evaluate_metrics(
                true_labels=test_labels,
                predicted_labels=gmm_labels,
                class_names=class_names,
                output_dir=gmm_output_dir
            )

            # Calculate and print silhouette score for GMM
            calculate_silhouette_score(test_data, gmm_labels, method_name="GMM", k=k)
            
            # Visualize GMM clusters if enabled
            if cluster_images:
                visualize_clusters(test_data, test_labels, gmm_labels, gmm_output_dir)
                save_sample_images(
                    data_dir=test_dir,
                    cluster_assignments=gmm_labels,
                    labels=test_labels,
                    output_dir=os.path.join(gmm_output_dir, "sample_images"),
                    image_size=(224, 224)
                )

            # K-Means Evaluation
            print(f"  Running K-Means with k={k}...")
            kmeans_labels = run_kmeans(test_data, n_clusters=k)

            kmeans_output_dir = os.path.join(output_dir, f"kmeans_k{k}")
            os.makedirs(kmeans_output_dir, exist_ok=True)

            # Evaluate metrics
            evaluate_metrics(
                true_labels=test_labels,
                predicted_labels=kmeans_labels,
                class_names=class_names,
                output_dir=kmeans_output_dir
            )

            # Calculate and print silhouette score for K-Means
            calculate_silhouette_score(test_data, kmeans_labels, method_name="K-Means", k=k)

            # Visualize K-Means clusters if enabled
            if cluster_images:
                visualize_clusters(test_data, test_labels, kmeans_labels, kmeans_output_dir)
                save_sample_images(
                    data_dir=test_dir,
                    cluster_assignments=kmeans_labels,
                    labels=test_labels,
                    output_dir=os.path.join(kmeans_output_dir, "sample_images"),
                    image_size=(224, 224)
                )

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run GMM and K-Means pipelines with PCA and clustering visualization')
    parser.add_argument('--train_dir', type=str, required=True, help='Path to training dataset')
    parser.add_argument('--test_dir', type=str, required=True, help='Path to test dataset')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save outputs')
    parser.add_argument('--k_values', type=int, nargs='+', default=[5, 10, 20], help='List of k values (number of clusters)')
    parser.add_argument('--covariance_type', type=str, nargs='+', default=['full'], choices=['full', 'tied', 'diag', 'spherical'], help='List of covariance types for GMM')
    parser.add_argument('--max_iters', type=int, default=10, help='Maximum number of iterations for GMM')
    parser.add_argument('--pca_components', type=int, default=50, help='Number of PCA components')
    parser.add_argument('--cluster_images', action='store_true', help='Enable clustering and visualization')

    args = parser.parse_args()
    main(
        args.train_dir,
        args.test_dir,
        args.output_dir,
        args.k_values,
        args.covariance_type,
        args.max_iters,
        args.pca_components,
        args.cluster_images
    )
