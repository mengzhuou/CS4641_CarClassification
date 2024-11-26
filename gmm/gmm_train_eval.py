import argparse
import os
import numpy as np
from datetime import datetime
from preprocess import preprocess_images
from gmm import GMM
from evaluate import evaluate_gmm
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import fowlkes_mallows_score


def flatten_images(data):
    """
    Flatten image data into a 2D array for PCA and GMM.
    Args:
        data (np.array): Image data (N, H, W, C).
    Returns:
        np.array: Flattened data (N, H*W*C).
    """
    return data.reshape(data.shape[0], -1)

def apply_pca(data, n_components=50):
    """
    Apply PCA to reduce dimensionality of the dataset.
    Args:
        data (np.array): Input data.
        n_components (int): Number of principal components to keep.
    Returns:
        np.array: Reduced data.
    """
    pca = PCA(n_components=n_components)
    data_reduced = pca.fit_transform(data)
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"PCA applied. Reduced data shape: {data_reduced.shape}, Explained Variance: {explained_variance:.2%}")
    return data_reduced

def train_gmm(data, K_values, max_iters=50, n_components=50):
    """
    Train GMM for multiple values of K.
    Args:
        data: Input data for GMM.
        K_values: List of K values (number of clusters).
        max_iters: Maximum iterations for GMM.
        n_components: Number of PCA components.
    Returns:
        dict: GMM results for each K.
    """
    data_flattened = flatten_images(data)  # Flatten images before PCA
    data_reduced = apply_pca(data_flattened, n_components)

    results = {}
    for K in K_values:
        print(f"\n===============================\nTraining GMM with K={K}\n===============================")
        gmm = GMM(data_reduced, K=K, max_iters=max_iters)
        tau, (pi, mu, sigma) = gmm(full_matrix=False)
        results[K] = {'tau': tau, 'pi': pi, 'mu': mu, 'sigma': sigma}

        # Compute log-likelihood for debugging
        joint_ll = gmm._ll_joint(pi, mu, sigma)
        log_likelihood = np.sum(gmm.logsumexp(joint_ll))
        print(f"K={K}, Log-Likelihood: {log_likelihood:.4f}")
    return results

def evaluate_gmm_with_sklearn(train_data, test_data, test_labels, n_components=48):
    # flatten and reduce dimensionality of train and test data
    train_data_flattened = flatten_images(train_data)
    test_data_flattened = flatten_images(test_data)
    
    # apply PCA
    train_data_reduced = apply_pca(train_data_flattened, n_components=n_components)
    test_data_reduced = apply_pca(test_data_flattened, n_components=n_components)

    # fit GMM on training data
    gmm = GaussianMixture(n_components=n_components, covariance_type='full', init_params='kmeans', max_iter=100)
    gmm.fit(train_data_reduced)
    
    # predict clusters
    train_labels_pred = gmm.predict(train_data_reduced)
    test_labels_pred = gmm.predict(test_data_reduced)
    
    # evaluate using Fowlkes-Mallows score
    score = fowlkes_mallows_score(test_labels, test_labels_pred)
    print(f"Fowlkes-Mallows Score for n_components={n_components}: {score}")
    return score

def main():
    parser = argparse.ArgumentParser(description="GMM Training and Evaluation")
    parser.add_argument('--train_dir', type=str, required=True, help="Path to training dataset")
    parser.add_argument('--test_dir', type=str, required=True, help="Path to test dataset")
    parser.add_argument('--output_dir', type=str, required=True, help="Base directory to save results")
    parser.add_argument('--max_iters', type=int, default=50, help="Maximum iterations for GMM")
    parser.add_argument('--K_values', type=int, nargs='+', default=[2, 3, 5, 10], help="List of K values for GMM")
    parser.add_argument('--n_components', type=int, default=50, help="Number of PCA components")
    args = parser.parse_args()

    # directory for storing results (unique name with current timestamp and max_iters)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_name = f"[{timestamp}]_{args.max_iters}-iters"
    output_dir = os.path.join(args.output_dir, unique_name)
    os.makedirs(output_dir, exist_ok=True)

    # preprocess datasets
    train_data, train_labels, _ = preprocess_images(args.train_dir)
    test_data, test_labels, _ = preprocess_images(args.test_dir)

    # train GMM with PCA
    gmm_results = train_gmm(train_data, args.K_values, max_iters=args.max_iters, n_components=args.n_components)

    # evaluate GMM with evaluate.py
    metrics = evaluate_gmm(test_data, test_labels, gmm_results)

    # evaluate GMM with sklearn GMM
    score = evaluate_gmm_with_sklearn(train_data, test_data, test_labels, n_components=args.n_components)

    # save results
    for K, result in gmm_results.items():
        # np.savez(os.path.join(output_dir, f'K_{K}_results.npz'), **result)
        with open(os.path.join(output_dir, f'K_{K}_metrics.txt'), 'w') as f:
            f.write(f"K={K}, Fowlkes-Mallows Score from evaluate.py: {metrics[K]:.4f}\n")
            f.write(f"K={K}, Fowlkes-Mallows Score for sklearn-based GMM evaluation: {score:.4f}\n")   

    print(f"Final Fowlkes-Mallows Score for sklearn-based GMM evaluation: {score:.4f}")            

    print(f"GMM training and evaluation completed. Results saved to {output_dir}")

if __name__ == "__main__":
    main()
