import torch
import torch.nn as nn
import torch.nn.functional as F

class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules, num_routes, in_channels, out_channels, kernel_size=None):
        super(CapsuleLayer, self).__init__()
        self.num_capsules = num_capsules
        self.num_routes = num_routes
        self.in_channels = in_channels
        self.out_channels = out_channels

        if kernel_size is not None:
            self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, kernel_size, kernel_size))
        else:
            self.W = nn.Parameter(torch.randn(1, num_routes, num_capsules, out_channels, in_channels))

    def forward(self, x, num_iterations=3):
        """
        Args:
            x: [B x InChannels x H x W] or [B x InFeatures]
        Returns:
            capsules: [B x NumCapsules x OutFeatures]
        """
        batch_size = x.size(0)
        if len(x.shape) == 4:
            x = x.view(batch_size, self.in_channels, -1)
            x = x.transpose(1, 2)

        u_hat = torch.matmul(self.W[:, :, :, :, :], x[:, None, :, :, None])
        u_hat = u_hat.squeeze(-1)

        b = torch.zeros(batch_size, self.num_routes, self.num_capsules, 1).to(x.device)

        for i in range(num_iterations):
            c = F.softmax(b, dim=2)
            s = (c * u_hat).sum(dim=1)
            v = self.squash(s)
            if i < num_iterations - 1:
                b = b + (u_hat * v.unsqueeze(1)).sum(dim=-1, keepdim=True)

        return v

    def squash(self, input_tensor):
        squared_norm = (input_tensor ** 2).sum(-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm)
        return scale * input_tensor / (torch.sqrt(squared_norm) + 1e-8)


class CapsuleDistillationNetwork(nn.Module):
    def __init__(self, feature_dim=64, num_capsules=16, hidden_dim=128):
        super().__init__()
        self.capsule_layer = CapsuleLayer(
            num_capsules=num_capsules,
            num_routes=feature_dim,
            in_channels=1,
            out_channels=hidden_dim
        )
        self.fc = nn.Linear(hidden_dim * num_capsules, feature_dim)

    def forward(self, x):
        """
        Args:
            x: [B x T x D] time-series features
        Returns:
            capsules: [B x D']
        """
        B, T, D = x.shape
        x = x.view(B, 1, T * D)
        x = x.unsqueeze(-1).unsqueeze(-1)  # Expand to fit capsule layer
        capsules = self.capsule_layer(x)
        return capsules.mean(dim=1)

    def capsule_alignment_loss(self, v_current, v_previous):
        """Compute L2 distance between current and previous capsule outputs"""
        return torch.norm(v_current - v_previous, p=2, dim=-1).mean()