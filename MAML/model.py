import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from meta_models import MetaModel, ClientMetaOptimizer
from hierarchical_task_embedding import HierarchicalTaskEmbedder
from secure_aggregation import SecureAggregator

class FMLightningModel(LightningModule):
    def __init__(self, base_model, num_classes=100, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['base_model'])

        # Base model and meta-model
        self.base_model = base_model
        self.meta_model = MetaModel(base_model)

        # Task Embedding
        self.task_embedder = HierarchicalTaskEmbedder(input_dim=64)

        # Secure Aggregation
        self.secure_agg = SecureAggregator(sigma=0.1)

        # Optimizer
        self.learning_rate = learning_rate
        self.client_optimizers = {}

    def forward(self, x):
        return self.meta_model(x)

    def training_step(self, batch, batch_idx):
        features = batch['features'].to(self.device)
        labels = batch['labels'].long().to(self.device)

        # Assume one support sample per class
        B, T, D = features.shape
        support_features = features[:, :10, :]
        support_labels = labels[:, :10]

        # Create task embedding
        task_embedding = self.task_embedder(features)

        # Simulate local adaptation
        adapted_model = MetaModel(self.base_model)
        adapted_model.copy_weights(self.base_model)
        with torch.enable_grad():
            output = adapted_model(support_features.view(B * 10, D))
            loss = F.cross_entropy(output, support_labels.view(-1))
            adapted_params = adapted_model.adapt(loss, lr=0.1)
            adapted_model.set_adapted_weights(adapted_params)

        # Evaluate on query set
        query_features = features[:, 10:, :].reshape(B * (T - 10), D)
        query_labels = labels[:, 10:].reshape(-1)
        logits = adapted_model(query_features)
        query_loss = F.cross_entropy(logits, query_labels)

        # Log metrics
        self.log('query_loss', query_loss, prog_bar=True)
        return query_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'query_loss'
        }