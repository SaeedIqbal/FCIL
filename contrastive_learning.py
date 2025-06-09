import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatioTemporalEncoder(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, 64, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        Args:
            x: [B x T x D]
        Returns:
            z: [B x D']
        """
        x = x.permute(0, 2, 1)  # [B x D x T]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        z = self.pool(x).squeeze(-1)  # [B x D']
        return z

class SpatioTemporalContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        """
        Compute spatio-temporal contrastive loss between two sequences.
        Args:
            z1: [B x D] embedding of first sequence
            z2: [B x D] embedding of second sequence
        Returns:
            loss: scalar
        """
        logits = torch.matmul(z1, z2.T) / self.temperature
        labels = torch.arange(logits.size(0)).to(z1.device)
        loss = F.cross_entropy(logits, labels)
        return loss

class SpatioTemporalContrastiveDetector:
    def __init__(self, device='cpu'):
        self.encoder = SpatioTemporalEncoder().to(device)
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-3)
        self.criterion = SpatioTemporalContrastiveLoss()
        self.device = device

    def detect_drift(self, x_prev, x_curr):
        """
        Detect drift via contrastive signal
        Args:
            x_prev: [T x D] previous sequence
            x_curr: [T x D] current sequence
        Returns:
            loss.item(): float
        """
        x_prev = torch.tensor(x_prev, dtype=torch.float32).unsqueeze(0).to(self.device)
        x_curr = torch.tensor(x_curr, dtype=torch.float32).unsqueeze(0).to(self.device)

        z_prev = self.encoder(x_prev)
        z_curr = self.encoder(x_curr)

        loss = self.criterion(z_prev, z_curr)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()