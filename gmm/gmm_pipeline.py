import os
import numpy as np
import argparse
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.metrics.cluster import fowlkes_mallows_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from gmm_model import GMM
from sklearn.decomposition import PCA

def load_and_flatten_data_subsample(data_dir, target_size=(224, 224), sample_size=None):
    datagen = ImageDataGenerator(rescale=1./255)
    dataset = datagen.flow_from_directory(data_dir, target_size=target_size, batch_size=1, shuffle=True)
    
    data = []
    labels = []
    for i, (img, label) in enumerate(dataset):
        data.append(img.flatten())
        labels.append(label)
        if sample_size and i >= sample_size - 1:
            break  # Stop after reaching the sample size
    
    return np.array(data), np.array(labels), dataset.class_indices

def reduce_dimensionality(X, n_components=100):
    """
    Reduces the dimensionality of the data using PCA.
    
    Args:
        X: The data to reduce, as a NumPy array (N x D).
        n_components: Number of principal components to retain.
        
    Returns:
        Reduced data, as a NumPy array (N x n_components).
    """
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(X)

def evaluate_gmm(model, X, y, class_names, output_dir, K):
    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)
    f1 = f1_score(y, predictions, average='weighted')
    fms = fowlkes_mallows_score(y, predictions)

    report = classification_report(y, predictions, target_names=class_names)
    conf_matrix = confusion_matrix(y, predictions)

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"classification_report_K{K}.txt")
    with open(report_path, "w") as f:
        f.write(report)

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

    args = parser.parse_args()

    # Load datasets
    train_data, train_labels, class_mapping = load_and_flatten_data_subsample(args.train_dir, sample_size=5000)
    test_data, test_labels, _ = load_and_flatten_data_subsample(args.test_dir, sample_size=5000)

    # Apply PCA for dimensionality reduction
    train_data = reduce_dimensionality(train_data, n_components=100)
    test_data = reduce_dimensionality(test_data, n_components=100)

    class_names = list(class_mapping.keys())

    # Loop over K values
    for K in args.K_values:
        try:
            print(f"\n\n\n===============================\nTraining GMM with K={K}\n===============================")
            gmm = GMM(num_components=K, max_iter=args.max_iters)
            gmm.fit(train_data)
            evaluate_gmm(gmm, test_data, test_labels, class_names, args.output_dir, K)
        except Exception as e:
            print(f"Failed to train GMM with K={K}: {e}")
