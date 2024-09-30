# utils/utils.py

import torch
import numpy as np
import torch.nn as nn
import random
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def init_bias_intersections(linear, isects):
    with torch.no_grad():
        W = linear.weight.data # [out_features, in_features]
        linear.bias.data = -W @ isects.to(linear.weight.device)

def plot_epoch_losses(epoch_losses):
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, marker='', linestyle='-', color='b')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.grid(True)
    plt.show()

def project_onto_decision_boundaries(pt, W, b):
    """
    Projects the mean point onto the decision boundary of each unit in the linear layer.
    
    Args:
        mean_point (torch.Tensor): A 1D tensor of shape (input_dim,) representing the mean point.
        linear_layer (torch.nn.Linear): The linear layer whose decision boundaries we are projecting onto.
    
    Returns:
        torch.Tensor: A tensor containing the projections of the mean point onto the decision boundary of each unit.
                      Shape is (output_dim, input_dim).
    """
    # Ensure `pt` is on the same device as the linear layer
    device = W.device
    pt = pt.to(device)

    # Ensure pt is a 1D tensor (a single point)
    if pt.dim() == 2 and pt.shape[0] == 1:
        pt = pt.squeeze(0)  # Convert [1, input_dim] to [input_dim]
    
    if pt.dim() != 1:
        raise ValueError(f"Point `pt` must be a 1D tensor, but got {pt.dim()} dimensions.")

    # Ensure the mean_point has the correct shape (input_dim,)
    if pt.shape[0] != W.shape[1]:
        raise ValueError("mean_point must have the same dimension as the input_dim of the linear layer.")
    
    # Calculate the projection of the mean point onto each decision boundary
    projections = []
    
    for i in range(W.shape[0]):  # Iterate over each unit
        w_i = W[i]  # Weights for the i-th unit (shape: input_dim,)
        b_i = b[i]  # Bias for the i-th unit (scalar)
        
        # Compute the projection of the mean point onto the decision boundary for unit i
        # The formula is: projection = mean_point - ((W * mean_point + b) / ||W||^2) * W
        w_norm_sq = torch.norm(w_i) ** 2  # ||W||^2
        projection = pt - ((w_i @ pt + b_i) / w_norm_sq) * w_i
        projections.append(projection)
    
    # Stack the projections for all units into a tensor
    return torch.stack(projections)  # Shape: (output_dim, input_dim)

def get_scaled_vectors(W):
    """
    Computes scaled vectors for each unit in the linear layer.
    The scaling is done by dividing the length of each weight vector by the square of its distance.
    
    Args:
        linear_layer (torch.nn.Linear): The linear layer whose weight vectors are to be scaled.
    
    Returns:
        torch.Tensor: A tensor containing the scaled vectors of the linear layer (shape: [output_dim, input_dim]).
    """
    # Compute the length (norm) of each weight vector (shape: [output_dim])
    lengths = torch.norm(W, dim=1, keepdim=True)
    
    # Compute the scale factor as length / (length ** 2)
    scale_factors = 1 / (lengths ** 2)
    
    # Scale the weight vectors (element-wise multiplication)
    scaled_vectors = W * scale_factors
    
    return scaled_vectors


# def combine_models(models):
#     """
#     Combines the linear layers of a list of models (each with a single node) into a single linear layer.
    
#     Args:
#         models (list): List of models, each containing a linear layer with a single node.
    
#     Returns:
#         nn.Linear: A new linear layer containing all the combined weights and biases.
#     """
#     # Extract the weights and biases from all the models
#     weights = []
#     biases = []
    
#     for model in models:
#         linear = model.linear  # Assuming each model has a single linear layer
#         weights.append(linear.weight.data)
#         biases.append(linear.bias.data)
    
#     # Combine weights and biases
#     combined_weights = torch.cat(weights, dim=0)  # Shape: (num_models, input_dim)
#     combined_biases = torch.cat(biases, dim=0)    # Shape: (num_models,)
    
#     # Create a new linear layer with combined weights and biases
#     input_dim = combined_weights.shape[1]
#     num_models = combined_weights.shape[0]
#     combined_linear = nn.Linear(input_dim, num_models)
    
#     # Manually set the weights and biases
#     with torch.no_grad():
#         combined_linear.weight.copy_(combined_weights)
#         combined_linear.bias.copy_(combined_biases)
    
#     return combined_linear
