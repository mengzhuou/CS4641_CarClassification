import os
import numpy as np
import argparse
from scipy.stats import mode
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.metrics.cluster import fowlkes_mallows_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.decomposition import PCA
from gmm_model import GMM


def load_and_flatten_data_subsample(data_dir, target_size=(224, 224), sample_size=None):
    """
    Load and flatten images from the given directory.
    Optionally subsamples the dataset for faster processing.
    """
    print("\nInside load_and_flatten_data_subsample()...")
    datagen = ImageDataGenerator(rescale=1./255)
    dataset = datagen.flow_from_directory(data_dir, target_size=target_size, batch_size=1, shuffle=True)
    
    data = []
    labels = []
    for i, (img, label) in enumerate(dataset):
        data.append(img.flatten())
        # Flatten the label from one-hot encoding to class index
        labels.append(np.argmax(label))  # Convert one-hot encoding to class index
        if sample_size and i >= sample_size - 1:
            break  # Stop after reaching the sample size
    
    return np.array(data), np.array(labels), dataset.class_indices

def cluster_images(data_dir, output_dir, K, max_iters=100, target_size=(224, 224), batch_size=1):
    """
    Clusters pixels in all images from the given directory and saves clustered images.

    Args:
        data_dir: Directory containing images.
        output_dir: Directory to save clustered images.
        K: Number of clusters/components for GMM.
        max_iters: Maximum number of iterations for GMM.
        target_size: Tuple for resizing images before clustering.
        batch_size: Number of images to process in a single batch.
    """
    os.makedirs(output_dir, exist_ok=True)
    datagen = ImageDataGenerator(rescale=1./255)
    dataset = datagen.flow_from_directory(data_dir, target_size=target_size, batch_size=batch_size, shuffle=False)
    
    # Process in batches
    for batch_idx, (batch_images, _) in enumerate(dataset):
        batch_size = batch_images.shape[0]
        flattened_pixels = batch_images.reshape(-1, batch_images.shape[-1])  # Flatten each batch

        # Train GMM on the batch
        gmm = GMM(flattened_pixels, K=K, max_iters=max_iters, init='kmeans')
        tau, (pi, mu, sigma) = gmm()  # Train GMM for this batch

        # Apply clustering and save each image in the batch
        for img_idx in range(batch_size):
            img_pixels = flattened_pixels[img_idx * (target_size[0] * target_size[1]):(img_idx + 1) * (target_size[0] * target_size[1])]
            clustered_pixels = mu[np.argmax(tau[img_idx * (target_size[0] * target_size[1]):(img_idx + 1) * (target_size[0] * target_size[1])], axis=1)]
            clustered_img = clustered_pixels.reshape(target_size + (batch_images.shape[-1],))
            
            # Save clustered image
            output_path = os.path.join(output_dir, f"clustered_image_{batch_idx}_{img_idx}.png")
            plt.imsave(output_path, clustered_img)

        # Stop if we finish processing all images
        if batch_idx * batch_size >= dataset.samples:
            break

def reduce_dimensionality(X, n_components=100):
    """
    Reduces the dimensionality of the data using PCA.
    Args:
        X: The data to reduce, as a NumPy array (N x D).
        n_components: Number of principal components to retain.
    Returns:
        Reduced data, as a NumPy array (N x n_components).
    """
    print(f"Reducing dimensionality to {n_components} components...")
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(X)

def map_clusters_to_labels(predictions, true_labels):
    """
    Maps GMM clusters to the closest true labels based on majority voting.
    Args:
        predictions: Cluster assignments (1D array).
        true_labels: Ground truth labels (1D array).
    Returns:
        remapped_predictions: Predictions mapped to true label space.
    """
    print("\nInside map_clusters_to_labels()")
    
    # Compute the confusion matrix
    conf_matrix = confusion_matrix(true_labels, predictions)

    # Find the most common true label for each cluster
    mapping = np.argmax(conf_matrix, axis=0)

    # Initialize mapping array
    mapping = np.zeros(conf_matrix.shape[1], dtype=int)
    
    # Find the most common true label for each cluster
    for cluster_idx in range(conf_matrix.shape[1]):
        if conf_matrix[:, cluster_idx].sum() > 0:
            mapping[cluster_idx] = np.argmax(conf_matrix[:, cluster_idx])
        else:
            mapping[cluster_idx] = -1  # Mark unmapped clusters

    # Remap predictions to the true label space
    remapped_predictions = np.array([
        mapping[cluster] if mapping[cluster] != -1 else mode(true_labels).mode[0]
        for cluster in predictions
    ])

    print(f"\nConfusion Matrix:\n{conf_matrix}")
    print(f"\nMapping of clusters to labels: {mapping}")

    return remapped_predictions

def evaluate_gmm(model, X, y, class_names, output_dir, K):
    """
    Evaluates the GMM model and saves results.
    """
    print("\nInside evaluate_gmm()...")
    # Convert one-hot encoded labels to class indices if necessary
    # if y.ndim > 1 and y.shape[1] > 1:
    #     print("Converting one-hot encoded labels to class indices.")
    #     y = np.argmax(y, axis=1)

    # get predictions from the model
    predictions = model.predict(X)

    # for debugging
    print("True labels (y):", y[:10])  # Display the first 10 labels
    print("Predictions:", predictions[:10])  # Display the first 10 predictions
    print("Unique classes in true labels:", np.unique(y))
    print("Unique classes in predictions:", np.unique(predictions))
    print("Type of labels:", type(y[0]))
    print("Type of predictions:", type(predictions[0]))

    # Map GMM clusters to true labels
    remapped_predictions = map_clusters_to_labels(predictions, y)

    # Ensure labels and predictions are integers
    if y.dtype != predictions.dtype:
        print("Converting labels and predictions to the same type.")
        y = y.astype(np.int32)
        predictions = predictions.astype(np.int32)

    # Validate shape and content
    assert y.shape == predictions.shape, "Mismatch in shape between labels and predictions"
    assert y.dtype == predictions.dtype, "Mismatch in dtype between labels and predictions"

    # calculate metrics
    accuracy = accuracy_score(y, predictions)
    f1 = f1_score(y, predictions, average='weighted')
    fms = fowlkes_mallows_score(y, predictions)

    # Generate and save classification report
    report = classification_report(y, predictions, target_names=class_names)
    conf_matrix = confusion_matrix(y, predictions)

    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"classification_report_K{K}.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # Plot and save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (K={K})')
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_K{K}.png"))
    plt.close()

    print(f"K={K}: Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Fowlkes-Mallows Score: {fms:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate GMM model.")
    parser.add_argument("--train_dir", type=str, required=True, help="Path to the training data.")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to the testing data.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save results.")
    parser.add_argument("--max_iters", type=int, default=100, help="Maximum number of iterations for GMM.")
    parser.add_argument("--K_values", type=int, nargs='+', default=[10, 15, 20], help="List of K values to test.")
    parser.add_argument("--n_components", type=int, default=100, help="Number of PCA components to reduce dimensionality.")
    parser.add_argument("--cluster_images", action='store_true', help="Flag to perform pixel clustering on images.")

    args = parser.parse_args()

    if args.cluster_images:
        clustered_images_dir = os.path.join(args.output_dir, "cluster_images")
        cluster_images(args.train_dir, clustered_images_dir, K=10, max_iters=args.max_iters)

    # Load datasets
    train_data, train_labels, class_mapping = load_and_flatten_data_subsample(args.train_dir, sample_size=5000)
    test_data, test_labels, _ = load_and_flatten_data_subsample(args.test_dir, sample_size=5000)

    print("\nAfter calling load_and_flatten_data_subsample()...")
    print(f"Train data shape: {train_data.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"First 10 train labels: {train_labels[:10]}")

    print(f"Test data shape: {test_data.shape}")
    print(f"Test labels shape: {test_labels.shape}")
    print(f"First 10 test labels: {test_labels[:10]}")

    # Apply PCA for dimensionality reduction
    train_data = reduce_dimensionality(train_data, n_components=args.n_components)
    test_data = reduce_dimensionality(test_data, n_components=args.n_components)

    class_names = list(class_mapping.keys())

    # Loop over K values
    for K in args.K_values:
        try:
            print(f"\n\n\n===============================\nTraining GMM with K={K}\n===============================")
            gmm = GMM(train_data, K=K, max_iters=args.max_iters, init='kmeans')
            tau, (pi, mu, sigma) = gmm()  # Train the GMM
            gmm.points = test_data  # Update the data for evaluation
            evaluate_gmm(gmm, test_data, test_labels, class_names, args.output_dir, K)
        except Exception as e:
            print(f"Failed to train GMM with K={K}: {e}")
