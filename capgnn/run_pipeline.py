import torch
from data_loader import get_dataloader
from capsule_network import CapsuleDistillationNetwork, CapsuleLayer
from gnn_hierarchy_model import GNNHierarchyModel
from hierarchical_attention import HierarchicalAttention, MultiLevelDistiller
from distillation_loss import CategoryGradientInducedDistillationLoss
from torch.utils.data import default_collate

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------------
# Initialize components
# ----------------------

base_model = CapsuleDistillationNetwork(feature_dim=64, num_capsules=16, hidden_dim=128).to(device)
capsule_distiller = CapsuleDistillationNetwork(feature_dim=64, num_capsules=16, hidden_dim=128).to(device)

num_classes = 100
gnn_hierarchy = GNNHierarchyModel(num_classes, embed_dim=64).to(device)
hierarchical_attn = HierarchicalAttention(num_layers=4, num_classes=num_classes).to(device)
distillation_loss = CategoryGradientInducedDistillationLoss(gamma=0.9).to(device)

# ----------------------
# Run pipeline on all domains
# ----------------------

domains = ['medical', 'robotics', 'edge']

for domain in domains:
    print(f"\n Running Hierarchical Distillation for domain: {domain}")
    loader = get_dataloader(domain)

    for i, sample in enumerate(loader):
        features = sample['features'][0].numpy()
        labels = sample['labels'][0].numpy()

        # temporal sequence
        seq_len = min(features.shape[0], 50)
        features = torch.tensor(features[:seq_len], dtype=torch.float32).unsqueeze(0).to(device)
        labels = torch.tensor(labels[:seq_len], dtype=torch.long).to(device)

        # previous task
        with torch.no_grad():
            old_gnn_output = gnn_hierarchy(labels)

        # Forward pass
        capsule_out = base_model(features)
        gnn_output = gnn_hierarchy(labels)
        hier_loss = hierarchical_attn([capsule_out], [old_gnn_output])

        print(f"[Hierarchical Attn] Loss: {hier_loss.item():.4f}")

        # Optional: Use GNN distillation loss
        gnn_loss = distillation_loss(capsule_out, old_gnn_output, labels, task_ids=torch.tensor([1]), task_history=[0])
        print(f"[GNN Distillation] Loss: {gnn_loss.item():.4f}")

        # Capsule alignment loss
        cap_loss = base_model.capsule_alignment_loss(capsule_out, old_gnn_output)
        print(f"[Capsule Alignment] Loss: {cap_loss.item():.4f}")