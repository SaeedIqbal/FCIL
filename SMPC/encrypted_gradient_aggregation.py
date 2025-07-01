import torch
from secure_computation import SecureAggregator

class EncryptedGradientCommunicator:
    def __init__(self, clients, aggregator=None):
        self.clients = clients
        self.aggregator = aggregator or SecureAggregator()

    def collect_gradients(self, model):
        """Collect gradients from all clients"""
        grads = []
        weights = []

        for client in self.clients:
            client_grad = self._flatten_grads(client.get_model_grads(model))
            grads.append(client_grad)
            weights.append(len(client.train_data))

        return grads, weights

    def aggregate_gradients(self, model):
        """Aggregate gradients using HE and DP"""
        grads, weights = self.collect_gradients(model)
        flat_avg_grad = self.aggregator.aggregate(grads, weights)

        # Reshape back to original model shape
        idx = 0
        for param in model.parameters():
            size = param.numel()
            grad_slice = flat_avg_grad[idx:idx+size].reshape(param.shape)
            param.grad = grad_slice
            idx += size

        return model

    def _flatten_grads(self, grads):
        """Flatten parameter gradients into a single tensor"""
        return torch.cat([g.view(-1) for g in grads])