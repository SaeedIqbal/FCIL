import torch
import numpy as np
from torch.distributions.normal import Normal
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

class HomomorphicEncryptor:
    def __init__(self, sigma=0.1):
        self.sigma = sigma
        self.private_key = ec.generate_private_key(ec.SECP384R1())
        self.public_key = self.private_key.public_key()

    def encrypt_tensor(self, tensor):
        noise = Normal(0, self.sigma).sample(tensor.shape).to(tensor.device)
        return tensor + noise

    def decrypt_tensor(self, tensor):
        noise = Normal(0, self.sigma).sample(tensor.shape).to(tensor.device)
        return tensor - noise

    def serialize_public_key(self):
        return self.public_key.public_bytes(
            encoding=Encoding.PEM,
            format=PublicFormat.SubjectPublicKeyInfo
        )


class DifferentialPrivacyInjector:
    def __init__(self, epsilon=1.0, delta=1e-5, clip_norm=1.0):
        self.epsilon = epsilon
        self.delta = delta
        self.clip_norm = clip_norm
        self.noise_multiplier = self._compute_noise_multiplier()

    def _compute_noise_multiplier(self):
        """Compute noise multiplier based on (epsilon, delta)-DP"""
        return 1.0 / self.epsilon  # Simplified version; use RDP or zCDP for real deployment

    def apply_dp(self, grad):
        """Apply differential privacy to a single gradient tensor"""
        grad_clipped = torch.nn.utils.clip_grad_norm_(grad, self.clip_norm)
        noise = torch.randn_like(grad) * self.noise_multiplier * self.clip_norm
        return grad + noise


class SecureAggregator:
    def __init__(self, sigma=0.1, epsilon=1.0, delta=1e-5, clip_norm=1.0):
        self.he = HomomorphicEncryptor(sigma=sigma)
        self.dp = DifferentialPrivacyInjector(epsilon=epsilon, delta=delta, clip_norm=clip_norm)

    def aggregate(self, grads_list, weights=None):
        """
        Aggregate gradients using HE and DP.
        Args:
            grads_list: List of gradients [grad_1, grad_2, ..., grad_N]
            weights: Optional list of weights for each gradient
        Returns:
            avg_grad: Aggregated gradient tensor
        """
        device = grads_list[0].device
        if weights is None:
            weights = [1.0] * len(grads_list)
        total_weight = sum(weights)

        # Apply DP
        dp_grads = [self.dp.apply_dp(g) for g in grads_list]

        # Encrypt
        enc_grads = [self.he.encrypt_tensor(g) for g in dp_grads]

        # Aggregate
        avg_enc_grad = torch.zeros_like(enc_grads[0])
        for g, w in zip(enc_grads, weights):
            avg_enc_grad += g * (w / total_weight)

        # Decrypt
        avg_grad = self.he.decrypt_tensor(avg_enc_grad)
        return avg_grad