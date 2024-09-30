# models/kmeans_classifier.py

from models.base_model import BaseModel
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

class KMeansClassifier(BaseModel):
    def __init__(self, num_clusters, input_dim):
        super(KMeansClassifier, self).__init__()
        self.num_clusters = num_clusters
        self.kmeans = KMeans(n_clusters=num_clusters)
        self.linear = nn.Linear(num_clusters, 10)  # Assuming 10 classes for MNIST

    def fit(self, X):
        # X is expected to be a NumPy array
        self.kmeans.fit(X)
        
    def forward(self, x):
        # x is a tensor
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1).cpu().numpy()
        distances = self.kmeans.transform(x_flat)  # Shape: [batch_size, num_clusters]
        distances = torch.from_numpy(distances).to(x.device)
        outputs = self.linear(distances)
        return outputs
