import numpy as np
from numpy.linalg import LinAlgError
from tqdm import tqdm
from kmeans import KMeans

SIGMA_CONST = 1e-06
LOG_CONST = 1e-32
FULL_MATRIX = False


class GMM(object):

    def __init__(self, X, K, max_iters=100):
        """
        Args:
            X: the observations/datapoints, N x D numpy array
            K: number of clusters/components
            max_iters: maximum number of iterations (used in EM implementation)
        """
        self.points = X
        self.max_iters = max_iters
        self.N = self.points.shape[0]
        self.D = self.points.shape[1]
        self.K = K

    def softmax(self, logit):
        """		
		Args:
		    logit: N x D numpy array
		Return:
		    prob: N x D numpy array. See the above function.
		Hint:
		    Add keepdims=True in your np.sum() function to avoid broadcast error.
		"""
        
        logit = np.clip(logit, -1e6, 1e6)  # to avoid overflow
        logit_stable = logit - np.max(logit, axis=1, keepdims=True)  # subtract max of each row to prevent num overflow
        exp_logit = np.exp(logit_stable)
        sum_exp_logit = np.sum(exp_logit, axis=1, keepdims=True)
        prob = exp_logit / sum_exp_logit
        prob = np.clip(prob, 1e-10, 1.0)  # prevent small probabilities

        return prob

    def logsumexp(self, logit):
        """		
		Args:
		    logit: N x D numpy array
		Return:
		    s: N x 1 array where s[i,0] = logsumexp(logit[i,:]). See the above function
		Hint:
		    The keepdims parameter could be handy
		"""
        
        max_logit = np.max(logit, axis=1, keepdims=True)
        s = max_logit + np.log(np.sum(np.exp(logit - max_logit), axis=1, keepdims=True))

        return s

    def normalPDF(self, points, mu_i, sigma_i):
        """		
		Args:
		    points: N x D numpy array
		    mu_i: (D,) numpy array, the center for the ith gaussian.
		    sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
		Return:
		    pdf: (N,) numpy array, the probability density value of N data for the ith gaussian
		
		Hint:
		    np.diagonal() should be handy.
		"""
        
        # get variances from sigma_i (covariance matrix)
        var = np.diagonal(sigma_i) + SIGMA_CONST

        # using log space to prevent overflow cause of large values
        diff = points - mu_i

        # exponent part of Gaussian PDF
        # -0.5 * (x - mu)^2 / variance
        log_exp_term = -0.5 * np.sum((diff ** 2) / var, axis=1)

        # coefficient part of the Gaussian distribution
        # 1 / sqrt(2 * pi * variance)
        log_coef = -0.5 * np.sum(np.log(2 * np.pi * var))

        # log PDF
        log_pdf = log_coef + log_exp_term

        # revert back to normal space
        pdf = np.exp(log_pdf)

        return np.clip(pdf, 1e-10, 1.0)

    def multinormalPDF(self, points, mu_i, sigma_i):
        """		
		Args:
		    points: N x D numpy array
		    mu_i: (D,) numpy array, the center for the ith gaussian.
		    sigma_i: DxD numpy array, the covariance matrix of the ith gaussian.
		Return:
		    normal_pdf: (N,) numpy array, the probability density value of N data for the ith gaussian
		
		Hint:
		    1. np.linalg.det() and np.linalg.inv() should be handy.
		    2. Note the value in self.D may be outdated and not correspond to the current dataset.
		    3. You may wanna check if the matrix is singular before implementing calculation process.
		"""

         # add small constant to sigma to avoid singular matrices
        sigma_i += SIGMA_CONST * np.eye(sigma_i.shape[0])
        D = mu_i.shape[0]  # DONT USE self.D

        try:
            # determinant and inverse of covariance matrix
            det_sigma = np.linalg.det(sigma_i)
            inv_sigma = np.linalg.inv(sigma_i)
        except LinAlgError:
            # try again if covariance matrix is stil numerically unstable
            det_sigma = np.linalg.det(sigma_i + SIGMA_CONST * np.eye(D))
            inv_sigma = np.linalg.inv(sigma_i + SIGMA_CONST * np.eye(D))

        # calculate PDF in log-space to improve stability
        diff = points - mu_i
        exp_term = -0.5 * np.sum(diff @ inv_sigma * diff, axis=1)
        coef = 1 / np.sqrt((2 * np.pi) ** D * det_sigma)
        pdf = coef * np.exp(exp_term)
        
        pdf = np.clip(pdf, 1e-10, 1.0)

        return pdf

    def create_pi(self):
        """		
		Initialize the prior probabilities
		Args:
		Return:
		pi: numpy array of length K, prior
		"""
        
        pi = np.ones(self.K) / (self.K)

        return pi

    def create_mu(self):
        """		
		Intialize random centers for each gaussian
		Args:
		Return:
		mu: KxD numpy array, the center for each gaussian.
		"""
        
        # randomly select K unique indices (initial cluster centers) from dataset of N points
        idx = np.random.choice(self.N, self.K, replace=True)
        # initialize mu
        mu = self.points[idx]

        return mu

    def create_sigma(self):
        """		
		Initialize the covariance matrix with np.eye() for each k. For grads, you can also initialize the
		by K diagonal matrices.
		Args:
		Return:
		sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
		    You will have KxDxD numpy array for full covariance matrix case
		"""
        
        # initializing covariance matrix for each K
        sigma = np.array([np.eye(self.D) * 1e-2 for _ in range(self.K)])

        return sigma

    def _init_components(self, **kwargs):
        """		
		Args:
		    kwargs: any other arguments you want
		Return:
		    pi: numpy array of length K, prior
		    mu: KxD numpy array, the center for each gaussian.
		    sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
		        You will have KxDxD numpy array for full covariance matrix case
		
		    Hint: np.random.seed(5) must be used at the start of this function to ensure consistent outputs.
		"""
        
        # np.random.seed(5)  # ensures consistent outputs

        # Use k-means to initialize means
        kmeans = KMeans(self.points, self.K, max_iters=100)
        kmeans.train()
        mu = kmeans.centers  # Use k-means centroids as initial means

        # Initialize priors (pi) uniformly
        pi = np.ones(self.K) / self.K

        # Compute covariances (sigma) based on k-means assignments
        sigma = []
        for k in range(self.K):
            cluster_points = self.points[kmeans.assignments == k]
            if len(cluster_points) > 0:
                cov_matrix = np.cov(cluster_points, rowvar=False)
                cov_matrix += np.eye(self.D) * SIGMA_CONST  # Regularize covariance matrix
                sigma.append(cov_matrix)
            else:
                sigma.append(np.eye(self.D) * SIGMA_CONST)  # Handle empty clusters
        sigma = np.array(sigma)

        return pi, mu, sigma

    def _ll_joint(self, pi, mu, sigma, full_matrix=FULL_MATRIX, **kwargs):
        """		
		Args:
		    pi: np array of length K, the prior of each component
		    mu: KxD numpy array, the center for each gaussian.
		    sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
		    array for full covariance matrix case
		    full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
		
		Return:
		    ll(log-likelihood): NxK array, where ll(i, k) = log pi(k) + log NormalPDF(points_i | mu[k], sigma[k])
		"""
        
        # NxK arr to store log-likelihood for each point-cluster pair
        ll = np.zeros((self.N, self.K))

        # loop over k Gaussian components
        for k in range(self.K):
            # probability density for each data point in the k-th component
            prob_density = self.normalPDF(self.points, mu[k], sigma[k])
            # log-likelihood of each point to be part of the k-th component
            ll[:,k] = np.log(pi[k] + LOG_CONST) + np.log(prob_density + LOG_CONST)  # LOG_CONST avoids having log(0)
        
        ll = np.clip(ll, -1e10, 1e10)  # clamp log-likelihood values
        
        return ll

    def _E_step(self, pi, mu, sigma, full_matrix=FULL_MATRIX, **kwargs):
        """		
		Args:
		    pi: np array of length K, the prior of each component
		    mu: KxD numpy array, the center for each gaussian.
		    sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.You will have KxDxD numpy
		    array for full covariance matrix case
		    full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
		Return:
		    tau: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
		
		Hint:
		    You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above.
		"""
        
        tau = self.softmax(self._ll_joint(pi, mu, sigma))
        # tau = np.clip(tau, 1e-10, 1.0)  # clamp values to prevent instability
        print(f"E-Step: Tau (first 5 rows): {tau[:5]}")
        return tau

    def _M_step(self, tau, full_matrix=FULL_MATRIX, **kwargs):
        """		
		Args:
		    tau: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
		    full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.
		Return:
		    pi: np array of length K, the prior of each component
		    mu: KxD numpy array, the center for each gaussian.
		    sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
		    array for full covariance matrix case
		
		Hint:
		    There are formulas in the slides and in the Jupyter Notebook.
		    Undergrads: To simplify your calculation in sigma, make sure to only take the diagonal terms in your covariance matrix
		"""
        
        # calculate sum of responsibilities for each component k
        Nk = np.sum(tau, axis=0)
        print(f"M-Step: Nk: {Nk}")
        
        # update current params 
        pi = Nk / self.N
        mu = np.dot(tau.T, self.points) / Nk[:,None]
        sigma = np.zeros((self.K, self.D, self.D))

        # update covariance matrix for each Gaussian
        for k in range(self.K):
            diff = self.points - mu[k]  # difference b/w points and mean for component k
            weighted_diff = tau[:,k][:,None] * diff  # weight difference by responsibilities (tau)
            sigma_k = np.dot(weighted_diff.T, diff) / Nk[k]  # calculate covariance matrix for component k
            sigma[k] = sigma_k + SIGMA_CONST * np.eye(self.D)  # only keep the diagonal elements
        
        return pi, mu, sigma

    def __call__(self, full_matrix=FULL_MATRIX, abs_tol=1e-16, rel_tol=
        1e-16, **kwargs):
        """		
		Args:
		    abs_tol: convergence criteria w.r.t absolute change of loss
		    rel_tol: convergence criteria w.r.t relative change of loss
		    kwargs: any additional arguments you want
		
		Return:
		    tau: NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
		    (pi, mu, sigma): (1xK np array, KxD numpy array, KxDxD numpy array)
		
		Hint:
		    You do not need to change it. For each iteration, we process E and M steps, then update the paramters.
		"""

        pi, mu, sigma = self._init_components(**kwargs)
        # progress bar for iterations
        pbar = tqdm(range(self.max_iters))

        prev_loss = None
        for it in pbar:
            # E-step: calculate posterior distribution (soft cluster assignment)
            tau = self._E_step(pi, mu, sigma, full_matrix)

            # M-step: update parameters (pi, mu, sigma)
            pi, mu, sigma = self._M_step(tau, full_matrix)

            # calculate the negative log-likelihood of observation
            joint_ll = self._ll_joint(pi, mu, sigma, full_matrix)
            loss = -np.sum(self.logsumexp(joint_ll))

            # Debug logging for key parameters
            print(f"Iteration {it}:")
            print(f"  Loss: {loss}")
            print(f"  Pi: {pi}")
            print(f"  Mu: {mu}")
            print(f"  Sigma: {sigma}")
            print(f"  Tau (first 5 rows): {tau[:5]}")  # get a subset of `tau` for readability

            # check convergence if it's not the first iteration
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            pbar.set_description("iter %d, loss: %.4f" % (it, loss))

        return tau, (pi, mu, sigma)
