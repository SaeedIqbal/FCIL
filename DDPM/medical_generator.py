import torch
import torch.nn as nn
from replay_utils import rand_augment

class MedicalImageGenerator(nn.Module):
    def __init__(self, diffusion_model):
        super().__init__()
        self.diffusion = diffusion_model

    def forward(self, shape, device='cpu'):
        return self.diffusion.sample(shape, device=device)

    def augment_sample(self, sample):
        return rand_augment(sample)