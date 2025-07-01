import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalAttention(nn.Module):
    def __init__(self, num_layers, num_classes, embed_dim=64):
        super(HierarchicalAttention, self).__init__()
        self.layer_weights = nn.Parameter(torch.ones(num_layers))
        self.class_weights = nn.Parameter(torch.ones(num_classes))

    def forward(self, h_list, old_h_list):
        """
        Args:
            h_list: List of tensors [B x T x D] per layer
            old_h_list: List of tensors [B x T x D] from previous task
        Returns:
            loss: Scalar distillation loss
        """
        total_loss = 0
        for l, (h, old_h) in enumerate(zip(h_list, old_h_list)):
            layer_weight = F.softmax(self.layer_weights, dim=0)[l]
            class_weighted_loss = 0
            for c in range(h.shape[-1]):
                class_weight = F.softmax(self.class_weights, dim=0)[c]
                kl = F.kl_div(F.log_softmax(h[..., c], dim=-1), F.softmax(old_h[..., c], dim=-1), reduction='batchmean')
                class_weighted_loss += class_weight * kl
            total_loss += layer_weight * class_weighted_loss
        return total_loss


class MultiLevelDistiller(nn.Module):
    def __init__(self, base_model, num_layers, num_classes):
        super(MultiLevelDistiller, self).__init__()
        self.base_model = base_model
        self.hier_attn = HierarchicalAttention(num_layers, num_classes)

    def extract_features(self, x):
        """Extract multi-layer features from base model"""
        feat_maps = []
        for module in self.base_model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                x = module(x)
                feat_maps.append(x)
        return feat_maps

    def forward(self, x, old_x):
        h_list = self.extract_features(x)
        old_h_list = self.extract_features(old_x)
        return self.hier_attn(h_list, old_h_list)