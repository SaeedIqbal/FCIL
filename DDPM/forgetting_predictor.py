import torch
import torch.nn as nn

class ForgettingPredictor(nn.Module):
    def __init__(self, feature_dim=64, num_heads=4, num_layers=2, num_classes=100):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, features):
        """
        Args:
            features: [B x T x D] time-series or batch features
        Returns:
            logits: [B x T x C] class logits
        """
        B, T, D = features.shape
        features = features.view(B * T, D)
        features = features.unsqueeze(0)  # Add dummy sequence dim
        encoded = self.transformer(features)
        encoded = encoded.squeeze(0).view(B, T, -1)
        logits = self.classifier(encoded)
        return logits

    def compute_forgetting_scores(self, features, labels):
        logits = self.forward(features)
        probs = F.softmax(logits, dim=-1)
        true_probs = probs.gather(2, labels.unsqueeze(-1)).squeeze()
        forgetting_scores = 1 - true_probs.mean(dim=1)
        return forgetting_scores