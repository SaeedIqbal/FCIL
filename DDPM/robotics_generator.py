import torch
from replay_utils import rand_augment

class RoboticObjectGenerator(nn.Module):
    def __init__(self, diffusion_model):
        super().__init__()
        self.diffusion = diffusion_model

    def generate(self, obs_vector, class_id, device='cpu'):
        """Generate object view conditioned on sensor input"""
        B, D = obs_vector.shape
        z = torch.randn((B, 64)).to(device)
        x_new = self.diffusion.sample(z.shape, device=device)
        return x_new

    def augment(self, x):
        return rand_augment(x)