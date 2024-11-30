import numpy as np
from tqdm import tqdm
from kmeans import KMeans

SIGMA_CONST = 1e-6
LOG_CONST = 1e-32


class GMM:
    def __init__(self, X, K, max_iters=100, init='kmeans'):
        """
        Args:
            X: The observations/datapoints, N x D numpy array.
            K: Number of clusters/components.
            max_iters: Maximum number of iterations for EM.
            init: Initialization method ('kmeans' or 'random').
        """
        print(f"\nInitializing GMM with {K} components and max_iters={max_iters}...")
        self.points = X
        self.max_iters = max_iters
        self.N, self.D = X.shape
        self.K = K
        self.init = init
        self.pi = None
        self.mu = None
        self.sigma = None

    def softmax(self, logit):
        logit_stable = logit - np.max(logit, axis=1, keepdims=True)
        exp_logit = np.exp(logit_stable)
        return exp_logit / np.sum(exp_logit, axis=1, keepdims=True)

    def logsumexp(self, logit):
        max_logit = np.max(logit, axis=1, keepdims=True)
        return max_logit + np.log(np.sum(np.exp(logit - max_logit), axis=1, keepdims=True))

    def normalPDF(self, points, mu_i, sigma_i):
        var = np.diagonal(sigma_i) + SIGMA_CONST
        exp = -0.5 * ((points - mu_i) ** 2) / var
        coef = 1 / np.sqrt(2 * np.pi * var)
        return np.prod(coef * np.exp(exp), axis=1)

    def create_pi(self):
        return np.ones(self.K) / self.K

    def create_mu(self):
        print("Initializing cluster centers.")
        if self.init == 'kmeans':
            # Initialize using KMeans
            print("Using KMeans initialization.")
            kmeans = KMeans(self.points, self.K)
            centers, _, _ = kmeans.train()
            # print(f"Initial cluster centers:\n{centers}\n")
            return centers
        elif self.init == 'random':
            print("Using random initialization.\n")
            # Randomly select points as initial centers
            idx = np.random.choice(self.N, self.K, replace=False)
            return self.points[idx]

    def create_sigma(self):
        return np.array([np.eye(self.D) for _ in range(self.K)])

    def _init_components(self):
        np.random.seed(5)
        return self.create_pi(), self.create_mu(), self.create_sigma()

    def _ll_joint(self, pi, mu, sigma):
        ll = np.zeros((self.N, self.K))
        for k in range(self.K):
            prob_density = self.normalPDF(self.points, mu[k], sigma[k])
            ll[:, k] = np.log(pi[k] + LOG_CONST) + np.log(prob_density + LOG_CONST)
        return ll

    def _E_step(self, pi, mu, sigma):
        return self.softmax(self._ll_joint(pi, mu, sigma))

    def _M_step(self, tau):
        Nk = np.sum(tau, axis=0)
        pi = Nk / self.N
        mu = np.dot(tau.T, self.points) / Nk[:, None]
        sigma = np.zeros((self.K, self.D, self.D))
        for k in range(self.K):
            diff = self.points - mu[k]
            weighted_diff = tau[:, k][:, None] * diff
            sigma[k] = np.dot(weighted_diff.T, diff) / Nk[k] + np.eye(self.D) * SIGMA_CONST
        return pi, mu, sigma

    def __call__(self, abs_tol=1e-16, rel_tol=1e-16):
        print(f"Initializing new GMM with K={self.K} and seed={np.random.get_state()[1][0]}")

        self.pi, self.mu, self.sigma = self._init_components()
        pbar = tqdm(range(self.max_iters))
        prev_loss = None
        
        for it in pbar:
            tau = self._E_step(self.pi, self.mu, self.sigma)
            self.pi, self.mu, self.sigma = self._M_step(tau)
            joint_ll = self._ll_joint(self.pi, self.mu, self.sigma)

            loss = -np.sum(self.logsumexp(joint_ll))
            if it and abs(prev_loss - loss) < abs_tol and abs(prev_loss - loss) / prev_loss < rel_tol:
                break
            prev_loss = loss

            pbar.set_description(f"iter {it}, loss: {loss:.4f}, change: {abs(prev_loss - loss):.6f}")

        return tau, (self.pi, self.mu, self.sigma)

    def predict(self, X):
        """
        Predicts the cluster assignments for each data point in X.
        Args:
            X: Data points to predict, a NumPy array of shape (N, D).
        Returns:
            cluster_assignments: A NumPy array of cluster labels (integers) of shape (N,).
        """
        # Update the points for prediction
        self.points = X

        # Calculate the posterior probabilities
        tau = self._E_step(self.pi, self.mu, self.sigma)

        # Handle edge cases where tau contains NaN or invalid values
        tau = np.nan_to_num(tau, nan=1.0/self.K)

        # Assign each point to the cluster with the highest posterior probability
        cluster_assignments = np.argmax(tau, axis=1)
        return cluster_assignments
