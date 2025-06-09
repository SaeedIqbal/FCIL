import numpy as np
from sklearn.mixture import BayesianGaussianMixture
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler

class HierarchicalDPMM:
    def __init__(self, n_components=10, window_size=100, threshold=0.75):
        self.bgmm = BayesianGaussianMixture(n_components=n_components, covariance_type='diag')
        self.scaler = StandardScaler()
        self.window_size = window_size
        self.threshold = threshold
        self.feature_buffer = []

    def update(self, features):
        """
        Update with new feature sequence and detect boundaries.
        Args:
            features: [T x d] numpy array of features
        Returns:
            boundary_detected: boolean
        """
        features = self.scaler.transform(features)
        self.feature_buffer.extend(features)

        if len(self.feature_buffer) > self.window_size:
            window = np.array(self.feature_buffer[-self.window_size:])
            self.bgmm.fit(window)
            log_probs = self.bgmm.score_samples(window)
            mean_log_prob = np.mean(log_probs)

            if hasattr(self, '_last_mean'):
                divergence = abs(mean_log_prob - self._last_mean)
                if divergence > self.threshold:
                    self.feature_buffer = []  # Reset buffer after boundary
                    self._last_mean = mean_log_prob
                    return True  # Task boundary detected
            else:
                self._last_mean = mean_log_prob
        return False  # No boundary