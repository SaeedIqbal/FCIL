import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionModel(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, num_steps=1000):
        super().__init__()
        self.num_steps = num_steps
        self.betas = torch.linspace(1e-4, 0.02, num_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x_0, t):
        """Forward process: add noise to input"""
        alpha_bar = self.alpha_bars[t].to(x_0.device)
        noise = torch.randn_like(x_0)
        mean = torch.sqrt(alpha_bar) * x_0
        var = 1 - alpha_bar
        x_t = mean + torch.sqrt(var) * noise
        return x_t, noise

    def reverse(self, x_t, t):
        """Reverse process: denoise from x_t to x_{t-1}"""
        if t > 0:
            noise_pred = self.model(x_t)
            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            x_t_prev = (x_t - (beta_t / torch.sqrt(1 - self.alpha_bars[t])) * noise_pred) / torch.sqrt(alpha_t)
            return x_t_prev
        else:
            return x_t

    @torch.no_grad()
    def sample(self, shape, device='cpu'):
        """Generate new samples from noise"""
        x_t = torch.randn(shape).to(device)
        for t in reversed(range(self.num_steps)):
            x_t = self.reverse(x_t, t)
        return x_t