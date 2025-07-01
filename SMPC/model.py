import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from encrypted_gradient_aggregation import EncryptedGradientCommunicator
from semantic_distiller import EncryptedSemanticDistiller

class PrivacyPreservingLightningModel(LightningModule):
    def __init__(self, base_model, num_clients=5, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters(ignore=['base_model'])

        # Base model
        self.base_model = base_model

        # Clients (simulate multiple clients)
        self.clients = [self.base_model for _ in range(num_clients)]

        # Secure aggregator
        self.secure_communicator = EncryptedGradientCommunicator(self.clients)

        # Distiller
        self.distiller = EncryptedSemanticDistiller(self.base_model)

        # Optimizer
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.base_model(x)

    def training_step(self, batch, batch_idx):
        features = batch['features'].to(self.device)
        labels = batch['labels'].long().to(self.device)

        # client-side update
        outputs = self.base_model(features)
        loss = F.cross_entropy(outputs, labels)

        # secure gradient communication
        encrypted_model = self.secure_communicator.aggregate_gradients(self.base_model)

        # Perform secure semantic distillation
        if batch_idx > 0:
            with torch.no_grad():
                prev_features = features.roll(shifts=1, dims=0)
            secure_loss = self.distiller.secure_mpc_distill(prev_features, features)
            loss += 0.2 * secure_loss

        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'train_loss'
        }