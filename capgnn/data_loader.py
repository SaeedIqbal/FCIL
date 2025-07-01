import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MedicalDataset(Dataset):
    def __init__(self, root_dir='/home/phd/datasets/medical/rare_diseases'):
        self.root_dir = root_dir
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True).item()
        features = data['features']
        labels = data.get('labels', None)
        return {'features': features, 'labels': labels, 'domain': 'medical'}

class RoboticsDataset(Dataset):
    def __init__(self, root_dir='/home/phd/datasets/robotics/object_sequences'):
        self.root_dir = root_dir
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True).item()
        features = data['features']
        labels = data.get('labels', None)
        return {'features': features, 'labels': labels, 'domain': 'robotics'}

class EdgeComputingDataset(Dataset):
    def __init__(self, root_dir='/home/phd/datasets/edge/user_activity'):
        self.root_dir = root_dir
        self.files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True).item()
        features = data['features']
        labels = data.get('labels', None)
        return {'features': features, 'labels': labels, 'domain': 'edge'}

def get_dataloader(domain='medical', batch_size=1):
    if domain == 'medical':
        dataset = MedicalDataset()
    elif domain == 'robotics':
        dataset = RoboticsDataset()
    elif domain == 'edge':
        dataset = EdgeComputingDataset()
    else:
        raise ValueError(f"Unsupported domain: {domain}")
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)