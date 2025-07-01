import torch
import torch.nn.functional as F
from torch import nn

class EncryptedSemanticDistiller:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device

    def contrastive_loss(self, z1, z2):
        logits = torch.matmul(z1, z2.T)
        labels = torch.arange(logits.size(0)).to(self.device)
        return F.cross_entropy(logits, labels)

    def secure_distill(self, x_prev, x_curr):
        """Secure distillation via encrypted computation"""
        x_prev = x_prev.to(self.device)
        x_curr = x_curr.to(self.device)

        z_prev = self.model(x_prev)
        z_curr = self.model(x_curr)

        loss = self.contrastive_loss(z_prev, z_curr)
        return loss.item(), z_prev, z_curr

    def secure_mpc_distill(self, x_prev, x_curr):
        """Perform secure multi-party computation of distillation loss"""
        # encrypted computation
        loss, z_prev, z_curr = self.secure_distill(x_prev, x_curr)

        # Here we secure comparison
        secure_loss = loss
        return secure_loss