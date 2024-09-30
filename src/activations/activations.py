# activations/activations.py

import torch.nn as nn
import torch

def get_activation(name):
    if name == 'ReLU':
        return nn.ReLU()
    elif name == 'Abs':
        return AbsActivation()
    else:
        raise ValueError(f"Unsupported activation function: {name}")

class AbsActivation(nn.Module):
    def forward(self, x):
        return torch.abs(x)
