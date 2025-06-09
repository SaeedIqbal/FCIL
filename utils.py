import os
import numpy as np

def load_domain_data(domain):
    dataloader = get_dataloader(domain)
    all_features = []
    for batch in dataloader:
        features = batch['features'][0].numpy()
        all_features.append(features)
    return all_features

def normalize_features(features):
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0) + 1e-6
    return (features - mean) / std