import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from capsule_network import CapsuleDistillationNetwork
from gnn_hierarchy_model import GNNHierarchyModel
from hierarchical_attention import MultiLevelDistiller
from distillation_loss import CategoryGradientInducedDistillationLoss

class FCILLightningModel(LightningModule):
    def __init__(self, num_classes=100, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # Base model
        self.capsule_model = CapsuleDistillationNetwork(feature_dim=64, num_capsules=16, hidden_dim=128)
        self.gnn_hierarchy = GNNHierarchyModel(num_classes=num_classes, embed_dim=64)

        # Distillation modules
        self.multi_level_distiller = MultiLevelDistiller(
            base_model=self.capsule_model,
            num_layers=4,
            num_classes=num_classes
        )
        self.distillation_loss = CategoryGradientInducedDistillationLoss(gamma=0.9)

        # Optimizer
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.capsule_model(x)

    def training_step(self, batch, batch_idx):
        features = batch['features'].to(self.device)
        labels = batch['labels'].long().to(self.device)

        # Simulate previous task output
        with torch.no_grad():
            old_output = self.gnn_hierarchy(labels)

        # Forward pass
        current_output = self.capsule_model(features)
        capsule_loss = self.capsule_model.capsule_alignment_loss(current_output, old_output)

        # GNN-based distillation
        gnn_output = self.gnn_hierarchy(labels)
        gnn_loss = F.kl_div(F.log_softmax(current_output, dim=-1), F.softmax(gnn_output, dim=-1), reduction='batchmean')

        # Multi-level hierarchical distillation
        hier_loss = self.multi_level_distiller(features, features)  # dummy input for demo

        # Total loss
        total_loss = 0.5 * capsule_loss + 0.3 * gnn_loss + 0.2 * hier_loss

        # Log metrics
        self.log('capsule_loss', capsule_loss, prog_bar=True)
        self.log('gnn_loss', gnn_loss, prog_bar=True)
        self.log('hierarchical_loss', hier_loss, prog_bar=True)
        self.log('total_loss', total_loss, prog_bar=True)

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'total_loss'
        }