import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GNNHierarchyModel(nn.Module):
    def __init__(self, num_classes, embed_dim=64, hidden_dim=128):
        super(GNNHierarchyModel, self).__init__()
        self.embed_dim = embed_dim
        self.class_embeddings = nn.Embedding(num_classes, embed_dim)

        # Assume identity graph (fully connected for now)
        self.edge_index = self.build_identity_graph(num_classes)

        self.conv1 = GCNConv(embed_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embed_dim)

    def build_identity_graph(self, n):
        """Build fully connected graph"""
        edge_index = []
        for i in range(n):
            for j in range(i + 1, n):
                edge_index.append([i, j])
                edge_index.append([j, i])
        return torch.tensor(edge_index, dtype=torch.long).T

    def forward(self, y_indices):
        """
        Args:
            y_indices: [B] tensor of class indices
        Returns:
            embeddings: [B x D] class embeddings
        """
        z = self.class_embeddings(y_indices)  # [B x D]
        z = self.conv1(z, self.edge_index)
        z = F.relu(z)
        z = self.conv2(z, self.edge_index)
        return z

    def hierarchical_distillation_loss(self, old_z, new_z):
        """KL divergence between old and new class representations"""
        old_probs = F.softmax(old_z, dim=-1)
        new_probs = F.log_softmax(new_z, dim=-1)
        return F.kl_div(new_probs, old_probs, reduction='batchmean')