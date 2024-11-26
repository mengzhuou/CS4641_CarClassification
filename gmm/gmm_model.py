import numpy as np
from scipy.stats import multivariate_normal

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
        self.means = X[np.random.choice(n_samples, self.num_components, replace=False)]
        self.covariances = np.array([np.cov(X, rowvar=False)] * self.num_components)
        self.weights = np.ones(self.num_components) / self.num_components

    def e_step(self, X):
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.num_components))
        for k in range(self.num_components):
            responsibilities[:, k] = self.weights[k] * multivariate_normal.pdf(X, self.means[k], self.covariances[k])
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities

    def m_step(self, X, responsibilities):
        n_samples, n_features = X.shape
        effective_counts = responsibilities.sum(axis=0)
        self.weights = effective_counts / n_samples
        self.means = np.dot(responsibilities.T, X) / effective_counts[:, np.newaxis]
        for k in range(self.num_components):
            diff = X - self.means[k]
            self.covariances[k] = np.dot((responsibilities[:, k][:, np.newaxis] * diff).T, diff) / effective_counts[k]

    def fit(self, X):
        self.initialize_parameters(X)
        log_likelihood = 0
        for iteration in range(self.max_iter):
            responsibilities = self.e_step(X)
            self.m_step(X, responsibilities)
            new_log_likelihood = np.sum(np.log(np.sum([
                self.weights[k] * multivariate_normal.pdf(X, self.means[k], self.covariances[k])
                for k in range(self.num_components)], axis=0)))
            if abs(new_log_likelihood - log_likelihood) < self.tol:
                break
            log_likelihood = new_log_likelihood

    def predict(self, X):
        responsibilities = self.e_step(X)
        return np.argmax(responsibilities, axis=1)
