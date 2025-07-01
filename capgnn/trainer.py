import pytorch_lightning as pl
from data_loader import get_dataloader
from model import FCILLightningModel

def train_fcil_model(domain='medical', max_epochs=10, batch_size=1):
    print(f"\n Starting PyTorch Lightning Training for Domain: {domain}")

    # Load dataset
    dataloader = get_dataloader(domain=domain, batch_size=batch_size)

    # Initialize Lightning Model
    model = FCILLightningModel(num_classes=100)

    # Define Trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=1,
        precision=16 if torch.cuda.is_available() else 32,
        enable_progress_bar=True,
        logger=pl.loggers.TensorBoardLogger(save_dir='./logs/', name=f'fcil_{domain}')
    )

    # Start Training
    trainer.fit(model=model, train_dataloaders=dataloader)

    # Save model
    save_path = f"./checkpoints/fcillightning_{domain}.pt"
    trainer.save_checkpoint(save_path)
    print(f"Model saved at: {save_path}")