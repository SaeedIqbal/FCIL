import torch
from torch import nn, optim
from pytorch_lightning import Trainer
from data_loader import get_dataloader
from ddpm_model import DDPM
from secure_computation import SecureAggregator
from forgetting_predictor import ForgettingPredictor
from meta_models import ClientMetaOptimizer
from model import DiffusionReplayLightningModel  # From Gap 5 implementation

class FederatedAveragingLoop:
    def __init__(self,
                 num_clients=5,
                 global_model=None,
                 client_domains=None,
                 num_classes=100,
                 rounds=10,
                 local_epochs=3,
                 batch_size=16,
                 learning_rate=1e-3):
        self.num_clients = num_clients
        self.rounds = rounds
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.client_domains = client_domains or ['medical'] * num_clients
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # Global model
        self.global_model = global_model or self._initialize_global_model()

        # Secure aggregator
        self.secure_agg = SecureAggregator(sigma=0.1, epsilon=1.0, delta=1e-5)

        # Client models
        self.client_models = [self._initialize_client_model() for _ in range(num_clients)]

        # Client-specific meta-optimizers
        self.client_optimizers = [ClientMetaOptimizer(meta_lr=self.learning_rate)
                                  for _ in range(num_clients)]

        # Forgetting predictor
        self.forgetter = ForgettingPredictor(feature_dim=64, num_classes=num_classes)

    def _initialize_global_model(self):
        return DiffusionReplayLightningModel(image_size=64, channels=3, num_classes=self.num_classes)

    def _initialize_client_model(self):
        return self._initialize_global_model()

    def _local_train(self, client_id, domain):
        """Train a client on its own dataset"""
        print(f"\n[CLIENT {client_id}] Training on domain: {domain}")

        # Load domain-specific data
        dataloader = get_dataloader(domain=domain, batch_size=self.batch_size)

        # Use PyTorch Lightning Trainer
        trainer = Trainer(
            max_epochs=self.local_epochs,
            accelerator="auto",
            devices="auto",
            logger=False,
            enable_progress_bar=True
        )

        # Train model
        trainer.fit(model=self.client_models[client_id], train_dataloaders=dataloader)

        # Return gradients
        grads = [p.grad.clone().cpu() for p in self.client_models[client_id].parameters()]
        weights = len(dataloader)
        return grads, weights

    def _aggregate(self, all_grads_and_weights):
        """Aggregate encrypted and differentially private gradients"""
        avg_grads = self.secure_agg.aggregate(all_grads_and_weights)
        idx = 0
        for param in self.global_model.parameters():
            param.grad = avg_grads[idx].to(param.device)
            idx += 1

    def _sync_client_models(self):
        """Update all client models with the latest global parameters"""
        for i in range(self.num_clients):
            self.client_models[i].load_state_dict(self.global_model.state_dict())

    def _compute_forgetting_scores(self, client_id, domain):
        """Use forgetting predictor to identify high-risk old categories"""
        dataloader = get_dataloader(domain=domain, batch_size=self.batch_size)
        for batch in dataloader:
            features = batch['features'].to('cpu')
            labels = batch['labels'].long().to('cpu')
            scores = self.forgetter.compute_forgetting_scores(features.unsqueeze(2), labels.unsqueeze(2))
            return scores.topk(k=10).indices.tolist()

    def run(self):
        """Run the full federated averaging loop with FCIL enhancements"""
        print("\nStarting Federated Class-Incremental Learning Loop\n")

        for round_num in range(self.rounds):
            print(f" Round {round_num + 1}/{self.rounds}")
            all_grads_and_weights = []

            for client_id in range(self.num_clients):
                domain = self.client_domains[client_id]
                print(f"🛠️ Client {client_id} ({domain}) is training...")
                grads, weight = self._local_train(client_id, domain)
                all_grads_and_weights.append((grads, weight))

                # Forgetting-aware prioritization
                high_risk_classes = self._compute_forgetting_scores(client_id, domain)
                print(f"Detected high-risk classes: {high_risk_classes}")

            # Aggregate securely
            print("Aggregating gradients with HE and DP")
            self._aggregate(all_grads_and_weights)

            # Update global model
            optimizer = optim.Adam(self.global_model.parameters(), lr=self.learning_rate)
            optimizer.step()
            optimizer.zero_grad()

            # Sync clients
            print(" Syncing client models with updated global model")
            self._sync_client_models()

            # Save checkpoint
            save_path = f"./checkpoints/global_model_round_{round_num + 1}.pt"
            torch.save(self.global_model.state_dict(), save_path)
            print(f" Model saved at: {save_path}")

        print("\n Federated Learning Completed Successfully")


def main():
    parser = argparse.ArgumentParser(description="Federated Class-Incremental Learning Trainer")
    parser.add_argument("--num_clients", type=int, default=5)
    parser.add_argument("--domains", nargs='+', default=['medical', 'robotics', 'edge', 'medical', 'edge'],
                        help="List of domains for each client (length must match num_clients)")
    parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    if len(args.domains) != args.num_clients:
        raise ValueError("Length of --domains must match --num_clients")

    # Initialize federated loop
    fed_avg_loop = FederatedAveragingLoop(
        num_clients=args.num_clients,
        client_domains=args.domains,
        num_classes=args.num_classes,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size
    )

    # Run training
    fed_avg_loop.run()


if __name__ == "__main__":
    main()