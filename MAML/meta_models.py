import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer

class MetaModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        return self.base_model(x)

    def copy_weights(self, model):
        """Copy weights from another model"""
        for target_param, source_param in zip(self.parameters(), model.parameters()):
            target_param.data.copy_(source_param.data)

    def adapt(self, loss, lr=0.01):
        """Perform a single gradient update on the current model"""
        grads = torch.autograd.grad(loss, self.parameters(), create_graph=True)
        adapted_params = [p - lr * g for p, g in zip(self.parameters(), grads)]
        return adapted_params

    def set_adapted_weights(self, adapted_params):
        """Set the adapted parameters"""
        for param, adapted in zip(self.parameters(), adapted_params):
            param.data.copy_(adapted.data)


class ClientMetaOptimizer:
    def __init__(self, meta_lr=1e-3):
        self.meta_lr = meta_lr

    def update(self, client_model, global_model, support_loss):
        """
        Update client model based on support loss.
        Args:
            client_model: Local model
            global_model: Global model
            support_loss: Loss computed on few-shot samples
        """
        grads = torch.autograd.grad(support_loss, client_model.parameters())
        for p_client, p_global, grad in zip(client_model.parameters(), global_model.parameters(), grads):
            p_client.data -= self.meta_lr * grad