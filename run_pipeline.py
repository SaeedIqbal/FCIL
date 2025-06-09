import torch
from data_loader import get_dataloader
from dpmm_model import HierarchicalDPMM
from contrastive_learning import SpatioTemporalContrastiveDetector, SpatioTemporalContrastiveLoss
from drift_detector import LightweightBayesianDriftDetector
from utils import normalize_features

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ----------------------
# Initialize components
# ----------------------

dpmm_detector = HierarchicalDPMM()
contrastive_detector = SpatioTemporalContrastiveDetector(device=device)
bayesian_drift = LightweightBayesianDriftDetector()

# ----------------------
# Run pipeline on all domains
# ----------------------

domains = ['medical', 'robotics', 'edge']

for domain in domains:
    print(f"\n🚀 Running pipeline for domain: {domain}")
    loader = get_dataloader(domain)

    for i, sample in enumerate(loader):
        features = sample['features'][0].numpy()
        features = normalize_features(features)

        # 1. DPMM-based dynamic task boundary detection
        boundary_dpmm = dpmm_detector.update(features)
        if boundary_dpmm:
            print(f"[DPMM] Detected task boundary at batch {i}")

        # 2. Spatio-temporal contrastive drift detection
        if i > 0:
            prev_features = sample_prev_features
            curr_features = features
            loss = contrastive_detector.detect_drift(prev_features, curr_features)
            print(f"[Contrastive] Drift score: {loss:.4f}")

        # 3. Lightweight Bayesian drift detection
        drift_score = bayesian_drift.update(features)
        if bayesian_drift.check_drift(features):
            print(f"[Bayesian] Detected significant drift with Mahalanobis score: {drift_score:.4f}")

        sample_prev_features = features