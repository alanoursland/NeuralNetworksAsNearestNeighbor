# models/feedforward_nn.py

from models.base_model import BaseModel
import torch.nn as nn
import torch.nn.functional as F

class FeedForwardNN(BaseModel):
    def __init__(self, input_dim, hidden_dim, output_dim, activation):
        super(FeedForwardNN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, output_dim),
            activation
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flattening the input
        distance_metric = self.layers(x)
        return -distance_metric
        # softmax_output = F.softmax(distance_metric, dim=1)
        # return 1 - softmax_output
