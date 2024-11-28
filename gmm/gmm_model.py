import numpy as np
from numpy.linalg import slogdet, inv
from tqdm import tqdm
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal

SIGMA_CONST = 1e-06

class GMM:
    def __init__(self, num_components, tol=1e-4, max_iter=100):
        self.num_components = num_components
        self.tol = tol
        self.max_iter = max_iter
        self.means = None
        self.covariances = None
        self.weights = None

    def initialize_parameters(self, X):
        n_samples, n_features = X.shape
        self.means = initialize_means(X, self.num_components)
        self.covariances = np.array([np.cov(X, rowvar=False)] * self.num_components)
        self.weights = np.ones(self.num_components) / self.num_components

    def e_step(self, X):
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.num_components))
        for k in range(self.num_components):
            try:
                responsibilities[:, k] = self.weights[k] * multivariate_gaussian(X, self.means[k], self.covariances[k])
            except np.linalg.LinAlgError:
                print(f"Covariance matrix at component {k} is not positive definite.")
                responsibilities[:, k] = SIGMA_CONST
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities

    def m_step(self, X, responsibilities):
        n_samples, n_features = X.shape
        effective_counts = responsibilities.sum(axis=0)
        self.weights = effective_counts / n_samples
        self.means = np.dot(responsibilities.T, X) / effective_counts[:, np.newaxis]

        for k in range(self.num_components):
            diff = X - self.means[k]
            print(f"Component {k}: determinant of covariance matrix = {np.linalg.det(self.covariances[k]):.6e}")
            cov_matrix = np.dot((responsibilities[:, k][:, np.newaxis] * diff).T, diff) / effective_counts[k]
            self.covariances[k] = cov_matrix + np.eye(n_features) * (SIGMA_CONST * np.mean(np.diag(cov_matrix)))

    def fit(self, X):
        self.initialize_parameters(X)
        log_likelihood = 0
        
        # progress bar for iterations
        pbar = tqdm(range(self.max_iter))

        for it in pbar:
            responsibilities = self.e_step(X)
            self.m_step(X, responsibilities)
            new_log_likelihood = np.sum(np.log(np.sum([
                self.weights[k] * multivariate_normal.pdf(X, self.means[k], self.covariances[k])
                for k in range(self.num_components)], axis=0)))
            
            # check convergence if it's not the first iteration
            if it:
                if abs(new_log_likelihood - log_likelihood) < self.tol:
                    break
            log_likelihood = new_log_likelihood
            pbar.set_description("iter %d, loss: %.4f" % (it, log_likelihood))

    def predict(self, X):
        responsibilities = self.e_step(X)
        return np.argmax(responsibilities, axis=1)


def multivariate_gaussian(X, mean, covariance):
    n_features = X.shape[1]
    det_sign, log_det = slogdet(covariance)
    precision = inv(covariance)
    diff = X - mean
    exp_term = np.sum(diff @ precision * diff, axis=1)
    return np.exp(-0.5 * exp_term) / np.sqrt((2 * np.pi)**n_features * det_sign * np.exp(log_det))

def initialize_means(X, K):
    kmeans = KMeans(n_clusters=K, random_state=42)
    kmeans.fit(X)
    return kmeans.cluster_centers_
