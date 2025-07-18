import pytorch_lightning as pl
from data_loader import get_dataloader
from model import PrivacyPreservingLightningModel
import argparse

def train_secure_fed_model(domain='medical', max_epochs=10, batch_size=1, num_clients=5):
    print(f"\n🔐 Starting Secure Federated Training for Domain: {domain}")

    # Load dataset
    dataloader = get_dataloader(domain=domain, batch_size=batch_size)

    # Dummy base model (can be replaced with ResNet, etc.)
    base_model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 100)
    )

    # Initialize Lightning Model
    model = PrivacyPreservingLightningModel(base_model=base_model, num_clients=num_clients)

    # Define Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=1,
        precision=16 if torch.cuda.is_available() else 32,
        enable_progress_bar=True,
        logger=pl.loggers.TensorBoardLogger(save_dir='./logs/', name=f'secure_fcil_{domain}')
    )

    # Start Training
    trainer.fit(model=model, train_dataloaders=dataloader)

    # Save model
    save_path = f"./checkpoints/secure_fcil_{domain}.pt"
    trainer.save_checkpoint(save_path)
    print(f"✅ Secure federated model saved at: {save_path}")