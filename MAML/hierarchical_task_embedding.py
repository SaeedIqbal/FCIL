import torch
import torch.nn as nn

class HierarchicalTaskEmbedder(nn.Module):
    def __init__(self, input_dim=64, hidden_dim=128, embed_dim=32):
        super(HierarchicalTaskEmbedder, self).__init__()
        self.task_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        """
        Encode features into task embeddings.
        Args:
            x: [B x T x D] time-series or batch features
        Returns:
            task_embeddings: [B x E]
        """
        B, T, D = x.shape
        x = x.view(B * T, D)
        embeddings = self.task_encoder(x)
        return embeddings.view(B, T, -1).mean(dim=1)