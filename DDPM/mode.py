import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from ddpm_model import DDPM
from generative_replay import DiffusionReplayer
from forgetting_predictor import ForgettingPredictor

class DiffusionReplayLightningModel(LightningModule):
    def __init__(self, image_size=32, channels=3, num_classes=100, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()

        # DDPM and replayer
        self.ddpm = DDPM(image_channels=channels, num_classes=num_classes)
        self.replayer = DiffusionReplayer(image_size=image_size, channels=channels, num_classes=num_classes)

        # Forgetting predictor
        self.forgetter = ForgettingPredictor(feature_dim=64, num_classes=num_classes)

        # Optimizer
        self.learning_rate = learning_rate

    def forward(self, x, t, y=None):
        return self.ddpm(x, t, y)

    def training_step(self, batch, batch_idx):
        features = batch['features'].to(self.device)
        labels = batch['labels'].long().to(self.device)

        # Simulate forgetting prediction
        forget_scores = self.forgetter.compute_forgetting_scores(features.unsqueeze(2), labels.unsqueeze(2))

        # Sample high-risk classes
        high_risk_classes = torch.where(forget_scores > 0.7)[0].unique().tolist()

        # Generate synthetic samples
        if len(high_risk_classes) > 0:
            replay_images, replay_labels = self.replayer.generate_replay_samples(high_risk_classes, batch_size=16)

            # Re-train local model on replay
            outputs = self.ddpm(replay_images, t=torch.randint(0, 1000, (len(replay_images),)))
            loss = F.mse_loss(outputs, replay_images)

            # Log metrics
            self.log('replay_loss', loss, prog_bar=True)
            return loss
        else:
            # No replay needed
            return torch.tensor(0.0, requires_grad=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'replay_loss'
        }