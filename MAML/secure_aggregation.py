import torch
import numpy as np
from torch.distributions.normal import Normal

def encrypt_tensor(tensor, sigma=0.1):
    noise = Normal(0, sigma).sample(tensor.shape).to(tensor.device)
    return tensor + noise

def decrypt_tensor(tensor, sigma=0.1):
    noise = Normal(0, sigma).sample(tensor.shape).to(tensor.device)
    return tensor - noise

class SecureAggregator:
    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def aggregate(self, gradients_list):
        """
        Aggregate encrypted gradients via FedAvg-style weighted average.
        Args:
            gradients_list: List of [(grads, weight), ...]
        Returns:
            avg_grads: Aggregated gradients
        """
        device = gradients_list[0][0].device
        total_weight = sum(w for _, w in gradients_list)
        avg_grads = None

        for grads, weight in gradients_list:
            encrypted_grads = encrypt_tensor(grads, self.sigma)
            if avg_grads is None:
                avg_grads = encrypted_grads * (weight / total_weight)
            else:
                avg_grads += encrypted_grads * (weight / total_weight)

        avg_grads = decrypt_tensor(avg_grads, self.sigma)
        return avg_grads