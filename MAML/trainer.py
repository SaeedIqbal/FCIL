import pytorch_lightning as pl
from data_loader import get_dataloader
from model import FMLightningModel
import torch
import argparse

def train_fml_model(domain='medical', max_epochs=10, batch_size=1, num_clients=5):
    print(f"\n Starting Federated Meta-Learning Training for Domain: {domain}")

    # Load dataset
    dataloader = get_dataloader(domain=domain, batch_size=batch_size)

    # Dummy base model (can be replaced with ResNet, etc.)
    base_model = nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 100)
    )

    # Initialize Lightning Model
    model = FMLightningModel(base_model=base_model)

    # Define Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=1,
        precision=16 if torch.cuda.is_available() else 32,
        enable_progress_bar=True,
        logger=pl.loggers.TensorBoardLogger(save_dir='./logs/', name=f'fml_{domain}')
    )

    # Start Training
    trainer.fit(model=model, train_dataloaders=dataloader)

    # Save model
    save_path = f"./checkpoints/fml_{domain}.pt"
    trainer.save_checkpoint(save_path)
    print(f"Model saved at: {save_path}")