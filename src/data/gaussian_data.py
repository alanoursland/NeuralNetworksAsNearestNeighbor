# data/gaussian_data.py

import torch
from torch.utils.data import Dataset

class GaussianDataset(Dataset):
    def __init__(self, num_samples=None, mean=None, cov=None, device='cpu', preloaded_data=None):
        self.device = device
        
        if preloaded_data is not None:
            # Load from preloaded data
            self.X = preloaded_data['X'].to(self.device)
            self.Y = preloaded_data['Y'].to(self.device)
            self.mean = preloaded_data['mean'].to(self.device)
            self.cov_inv = preloaded_data['cov_inv'].to(self.device)
        else:
            # Generate new samples from a 2D Gaussian distribution
            dist = torch.distributions.MultivariateNormal(mean, cov)
            self.X = dist.sample((num_samples,)).to(self.device)

            # Compute Mahalanobis distances
            self.cov_inv = torch.inverse(cov).to(self.device)
            self.mean = mean.to(self.device)
            self.Y = self.mahalanobis_distance(self.X).unsqueeze(1)
        
    def mahalanobis_distance(self, x):
        delta = x - self.mean
        m_dist = torch.sqrt(torch.sum(delta @ self.cov_inv * delta, dim=1))
        return m_dist

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
