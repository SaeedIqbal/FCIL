import torch
import torch.nn as nn

class EdgeUserActivityGenerator(nn.Module):
    def __init__(self, diffusion_model):
        super().__init__()
        self.diffusion = diffusion_model

    def generate(self, class_id, device='cpu', low_rank=True):
        """Generate low-rank representation for on-device replay"""
        z = torch.randn((1, 32 if low_rank else 64)).to(device)
        x = self.diffusion.sample(z.shape, device=device)
        return x

    def replay_loss(self, model, x_gen, y_gen):
        logits = model(x_gen)
        return F.cross_entropy(logits, y_gen)