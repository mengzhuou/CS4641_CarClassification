import numpy as np
from kmeans import fowlkes_mallow


def evaluate_gmm(data, labels, gmm_results):
    """
    Evaluate GMM results using Fowlkes-Mallows score.
    
    Args:
        data (numpy array): Preprocessed data.
        labels (numpy array): Ground truth labels.
        gmm_results (dict): GMM results for each K.
        
    Returns:
        dict: Fowlkes-Mallows score for each K.
    """
    metrics = {}
    for K, result in gmm_results.items():
        cluster_assignments = np.argmax(result['tau'], axis=1)
        score = fowlkes_mallow(labels, cluster_assignments)
        metrics[K] = score
        print(f"K={K}, Fowlkes-Mallows Score: {score:.4f}")
    return metrics
