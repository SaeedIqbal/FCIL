import numpy as np
from scipy.stats import multivariate_normal
from sklearn.metrics import mahalanobis

class LightweightBayesianDriftDetector:
    def __init__(self, history_length=50):
        self.mu_history = []
        self.sigma_history = []
        self.history_length = history_length

    def update(self, features):
        """
        Update the drift detector with new features
        Args:
            features: [T x D] numpy array
        Returns:
            drift_score: float
        """
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)

        if len(self.mu_history) > 0:
            last_mu = self.mu_history[-1]
            last_sigma = self.sigma_history[-1]
            inv_last_sigma = np.linalg.pinv(last_sigma)
            diff = mu - last_mu
            score = np.sqrt(np.dot(diff.T, np.dot(inv_last_sigma, diff)))
        else:
            score = 0

        self.mu_history.append(mu)
        self.sigma_history.append(sigma)

        if len(self.mu_history) > self.history_length:
            self.mu_history.pop(0)
            self.sigma_history.pop(0)

        return score

    def check_drift(self, features, threshold=3.0):
        score = self.update(features)
        return score > threshold