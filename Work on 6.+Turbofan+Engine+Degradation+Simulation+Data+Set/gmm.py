"""
Gaussian Mixture Models (GMM) trained via Expectation-Maximization (EM).
Notation aligns with Stanford CS229 Lecture Notes (Andrew Ng):
  - phi_j: Mixture weights P(z = j)
  - mu_j: Mean vectors
  - sigma_j: Covariance matrices
  - w_j^(i): Responsibilities P(z^(i) = j | x^(i))
"""

import numpy as np


class GaussianMixture(object):
    """
    Expectation-Maximization algorithm for Gaussian Mixture Models (GMM)
    using only NumPy.

    Parameters
    ----------
    k : int, default=3
        Number of mixture components (Gaussians).
    max_iter : int, default=100
        Maximum number of EM iterations.
    eps : float, default=1e-4
        Convergence threshold for log-likelihood difference.
    reg_cov : float, default=1e-6
        Regularization added to the diagonal of covariances to prevent singularity.
    random_state : int, optional
        Seed for reproducibility.
    """
    def __init__(self, k=3, max_iter=100, eps=1e-4, reg_cov=1e-6, random_state=None):
        self.k = k
        self.max_iter = max_iter
        self.eps = eps
        self.reg_cov = reg_cov
        self.random_state = random_state

        # Parameters to be estimated
        self.phi_ = None    # shape (k,)
        self.mu_ = None     # shape (k, n_features)
        self.sigma_ = None  # shape (k, n_features, n_features)

    def _log_gaussian(self, X, mu, sigma):
        """
        Compute log N(x | mu, sigma) for every row in X.
        """
        n_features = X.shape[1]
        diff = X - mu

        sign, log_det = np.linalg.slogdet(sigma)
        if sign <= 0:
            raise ValueError("Covariance matrix sigma must be positive definite.")

        # Solve sigma * z = diff.T instead of computing explicit inverse
        solved = np.linalg.solve(sigma, diff.T).T
        mahalanobis = np.sum(diff * solved, axis=1)

        return -0.5 * (n_features * np.log(2.0 * np.pi) + log_det + mahalanobis)

    @staticmethod
    def _logsumexp(A, axis=1):
        """Numerically stable log(sum(exp(A)))."""
        max_A = np.max(A, axis=axis, keepdims=True)
        result = max_A + np.log(np.sum(np.exp(A - max_A), axis=axis, keepdims=True))
        return np.squeeze(result, axis=axis)

    def _estimate_log_prob(self, X):
        """Log joint probability: log(phi_j * N(x | mu_j, sigma_j))."""
        log_prob = np.empty((X.shape[0], self.k))
        for j in range(self.k):
            log_prob[:, j] = np.log(self.phi_[j]) + self._log_gaussian(X, self.mu_[j], self.sigma_[j])
        return log_prob

    def _e_step(self, X):
        """
        E-step: calculate posterior responsibilities:
        w_j^(i) := p(z^(i) = j | x^(i); phi, mu, sigma)
        """
        log_joint = self._estimate_log_prob(X)
        log_norm = self._logsumexp(log_joint, axis=1)

        log_w = log_joint - log_norm[:, None]
        w = np.exp(log_w)  # w_j^(i)
        avg_log_likelihood = np.mean(log_norm)

        return w, avg_log_likelihood

    def _m_step(self, X, w):
        """
        M-step: update parameters phi, mu, and sigma based on weights w.
        """
        n_samples, n_features = X.shape

        sum_w = w.sum(axis=0) + 1e-10

        # phi_j := 1/m * sum_i w_j^(i)
        self.phi_ = sum_w / n_samples

        # mu_j := sum_i (w_j^(i) * x^(i)) / sum_i w_j^(i)
        self.mu_ = (w.T @ X) / sum_w[:, None]

        # sigma_j := sum_i w_j^(i) (x^(i) - mu_j)(x^(i) - mu_j)^T / sum_i w_j^(i)
        self.sigma_ = np.empty((self.k, n_features, n_features))
        for j in range(self.k):
            diff = X - self.mu_[j]
            cov = ((diff * w[:, j:j+1]).T @ diff) / sum_w[j]
            # Regularization to prevent singular covariance
            cov += self.reg_cov * np.eye(n_features)
            self.sigma_[j] = cov

    def fit(self, X):
        """
        Fit the Gaussian Mixture Model to data X via EM algorithm.
        """
        X = np.asarray(X, dtype=float)

        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("X must be a non-empty 2D array.")
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values.")
        if not 1 <= self.k <= X.shape[0]:
            raise ValueError("k must be between 1 and n_samples.")

        rng = np.random.default_rng(self.random_state)
        n_samples, n_features = X.shape

        # Initialize mu from randomly selected data points
        indices = rng.choice(n_samples, size=self.k, replace=False)
        self.mu_ = X[indices].copy()

        # Initialize uniform phi: phi_j = 1/k
        self.phi_ = np.full(self.k, 1.0 / self.k)

        # Initialize shared sigma from data covariance
        global_cov = np.atleast_2d(np.cov(X, rowvar=False)) + self.reg_cov * np.eye(n_features)
        self.sigma_ = np.repeat(global_cov[None, :, :], self.k, axis=0)

        self.lower_bound_ = -np.inf
        self.converged_ = False
        self.n_iter_ = 0
        self.log_likelihood_history_ = []

        # EM iterations
        for i in range(self.max_iter):
            # E-step
            w, log_likelihood = self._e_step(X)
            self.log_likelihood_history_.append(log_likelihood)

            # Check convergence
            if i > 0:
                improvement = log_likelihood - self.lower_bound_
                if abs(improvement) < self.eps:
                    self.converged_ = True
                    self.n_iter_ = i
                    break

            self.lower_bound_ = log_likelihood

            # M-step
            self._m_step(X, w)
            self.n_iter_ = i + 1

        # Final likelihood evaluation
        w, self.lower_bound_ = self._e_step(X)
        self.log_likelihood_history_.append(self.lower_bound_)

        return self

    def predict_proba(self, X):
        """Posterior probabilities w_j^(i) for each sample."""
        X = np.asarray(X, dtype=float)
        w, _ = self._e_step(X)
        return w

    def predict(self, X):
        """Return the most likely component (cluster assignment) for each sample."""
        return np.argmax(self.predict_proba(X), axis=1)

    def score_samples(self, X):
        """Return log p(x) for each sample."""
        X = np.asarray(X, dtype=float)
        return self._logsumexp(self._estimate_log_prob(X), axis=1)

    def score(self, X):
        """Return average log-likelihood across all samples."""
        return np.mean(self.score_samples(X))
