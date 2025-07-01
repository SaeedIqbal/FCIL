import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms

def rand_augment(x, strength=0.5):
    """Apply random augmentation to generated images"""
    transform = transforms.Compose([
        transforms.RandomAffine(degrees=int(30 * strength), translate=(0.1 * strength, 0.1 * strength)),
        transforms.ColorJitter(brightness=strength, contrast=strength),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.2), ratio=(0.3, 1.7))
    ])
    return transform(x)

class ForgettingAwareSampler:
    def __init__(self, memory_size=100, forget_threshold=0.2):
        self.memory = {}
        self.memory_size = memory_size
        self.forget_threshold = forget_threshold

    def update_forgetting_scores(self, class_scores):
        """Update forgetting probabilities per class"""
        for cls, score in class_scores.items():
            if cls not in self.memory:
                self.memory[cls] = []
            self.memory[cls].append(score)
            if len(self.memory[cls]) > self.memory_size:
                self.memory[cls].pop(0)

    def get_priority_classes(self):
        """Return classes with highest forgetting risk"""
        priorities = {
            cls: np.mean(scores)
            for cls, scores in self.memory.items()
            if np.mean(scores) > self.forget_threshold
        }
        return sorted(priorities.items(), key=lambda x: x[1], reverse=True)

    def generate_samples(self, diffusion_model, class_ids, device='cpu'):
        """Generate synthetic samples for high-risk classes"""
        samples = []
        for cls, _ in self.get_priority_classes():
            z = torch.randn((1, 64)).to(device)
            sample = diffusion_model.sample(z.shape, device=device)
            samples.append((sample, cls))
        return samples