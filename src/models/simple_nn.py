# models/simple_nn.py

from models.base_model import BaseModel
import torch.nn as nn

class SimpleNN(BaseModel):
    def __init__(self, input_dim, activation):
        super(SimpleNN, self).__init__()
        self.linear = nn.Linear(input_dim, 1, bias=True)
        self.activation = activation
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        out = self.linear(x)
        out = self.activation(out)
        return out
