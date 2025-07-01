import torch
import torch.nn as nn
from torchvision import transforms
from ddpm_model import DDPM

class DiffusionReplayer(nn.Module):
    def __init__(self, image_size=32, channels=3, num_classes=100, augment=True):
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        self.num_classes = num_classes
        self.augment = augment

        # DDPM model
        self.ddpm = DDPM(image_channels=channels, num_classes=num_classes)

        # Augmentation pipeline
        self.aug_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1),
            transforms.RandomRotation(10)
        ]) if augment else lambda x: x

    def forward(self, x, t, y=None):
        return self.ddpm(x, t, y)

    def sample(self, y=None, batch_size=16, device='cpu'):
        """Generate images conditioned on class label y"""
        shape = (batch_size, self.channels, self.image_size, self.image_size)
        x = torch.randn(shape, device=device)

        for t in reversed(range(self.ddpm.n_steps)):
            with torch.no_grad():
                t_batch = torch.tensor([t]).repeat(batch_size).to(device)
                x = self.ddpm(x=x, t=t_batch, y=y)

        return x

    def generate_replay_samples(self, known_classes, batch_size=16):
        """Generate virtual samples for known classes"""
        device = next(self.parameters()).device
        replay_samples = []

        for cls in known_classes:
            y = torch.tensor([cls] * batch_size, device=device)
            samples = self.sample(y=y, batch_size=batch_size, device=device)
            replay_samples.append(samples)

        return torch.cat(replay_samples, dim=0), torch.repeat_interleave(
            torch.tensor(known_classes, device=device), repeats=batch_size
        )


class LatentSpacePerturbator:
    def __init__(self, sigma=0.1):
        self.sigma = sigma

    def perturb_latent(self, z):
        """Add Gaussian noise to latent representations"""
        noise = torch.randn_like(z) * self.sigma
        return z + noise