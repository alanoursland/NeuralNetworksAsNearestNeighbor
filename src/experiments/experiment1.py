# experiments/experiment1.py

import os
import torch
from data.gaussian_data import GaussianDataset
from models.simple_nn import SimpleNN
from activations.activations import get_activation
from training.trainer import Trainer
from utils.utils import init_bias_intersections
from utils.utils import plot_epoch_losses
from utils.utils import project_onto_decision_boundaries
from utils.utils import get_scaled_vectors
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

def plot_ex1_results(
        x, 
        means, dirs, classes,
        title="Learned Weight Vectors on 2D Gaussian Distribution", 
        xlabel="Feature 1", ylabel="Feature 2", 
        filename=None):
    """
    Plots a 2D point cloud from a tensor of shape [n, 2] and highlights means and direction vectors.
    
    Args:
        x (torch.Tensor): Tensor of shape [n, 2] containing the main 2D points to plot.
        means (torch.Tensor): Tensor of shape [m, 2] containing the mean points where vectors are drawn.
        dirs (torch.Tensor): Tensor of shape [m, 2] containing direction vectors to be drawn from each mean.
        title (str): Title of the plot.
        xlabel (str): Label for the x-axis.
        ylabel (str): Label for the y-axis.
        filename (str): Optional filename to save the figure (default: None, won't save).
    """
    if x.shape[1] != 2:
        raise ValueError("Input tensor must be of shape [n, 2] for 2D point plotting.")
    
    # Convert the tensors to numpy for compatibility with matplotlib
    points_np = x.cpu().numpy()
    means_np = means.cpu().numpy()
    dirs_np = dirs.cpu().numpy()

    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot main points
    plt.scatter(points_np[:, 0], points_np[:, 1], c='#ADD8E6', s=5, alpha=0.6, label="Data points")
    
    class_styles = [
        {'color': 'blue', 'name':'Coherent'},
        {'color': 'red', 'name':'Adhoc'},
        {'color': 'violet', 'name':'Inactive'},
    ]
    # Draw the direction vectors at each mean point
    # for class_id in range(1,2):
    for class_id in range(classes.max().item() + 1):
        print(class_id)
        style = class_styles[class_id]
        label = style["name"]
        color = style["color"]
        for i, mean in enumerate(means_np):
            if class_id == classes[i].item():
                dir_vector = dirs_np[i]
                plt.arrow(mean[0], mean[1], dir_vector[0], dir_vector[1], 
                        head_width=0.2, head_length=0.3, 
                        fc=color, ec=color, 
                        label=label,
                        length_includes_head=True, 
                        width=0.02, 
                        alpha=0.5)
                label = None # Only label the first one

    # Set labels and adjust plot
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.legend(loc="upper right")

    # Add scale bar for weight vectors
    plt.arrow(8, -9, 1, 0, head_width=0.2, head_length=0.3, fc='black', ec='black', length_includes_head=True)
    plt.text(8.5, -8.5, f"Unit vector", ha='center', va='bottom', fontsize=8)

    plt.tight_layout()

    # Show or save the plot
    if filename:
        plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')
    else:
        plt.show()

    plt.close()

def mean_centered_initialization(model, direction_range=(0, 2*np.pi), length_range=(0.1, 2.0), b_range=(-10.0, 10.0)):
    with torch.no_grad():
        # Sample a radian direction (theta) uniformly from the given range
        theta = np.random.uniform(*direction_range)
        
        # Sample a vector length uniformly from the given range
        length = np.random.uniform(*length_range)
        
        # Compute the components of the weight vector based on direction and length
        W_x = np.cos(theta) * length
        W_y = np.sin(theta) * length
        
        # Set the model's weights
        model.linear.weight.data[0, 0] = W_x
        model.linear.weight.data[0, 1] = W_y
        
        # Sample and set the bias uniformly from the given range
        model.linear.bias.data.uniform_(*b_range)

import torch

def classify_metrics(metric, thresholds=[]):
    """
    Args:
        metric (torch.Tensor): 1D tensor of metric values for each vector.
        thresholds (list or torch.Tensor): List of threshold values sorted in ascending order.
    """
    thresholds = torch.tensor(thresholds, device=metric.device).unsqueeze(0)  # Shape: [1, n]
    metric = metric.unsqueeze(1)  # Shape: [m, 1]
    classes = (metric > thresholds).sum(dim=1).long()
    return classes

def hyperplane_distances(x, weights, biases):
    with torch.no_grad():  # Disable gradient computation for evaluation
        x = x.unsqueeze(0) # [1, 2]
        linear_output = torch.matmul(weights, x.T).squeeze(1) + biases  # Shape: [m]
        results = torch.abs(linear_output)  # Shape: [m]
    return results

def run_experiment1(device, num_samples=3000, num_models=1000, num_epochs=100, activation_name='ReLU', load_existing=False):
    # Define results directory
    results_dir = f"results/ex1/{activation_name}-data_sampled"
    os.makedirs(results_dir, exist_ok=True)

    # Paths for saved data
    dataset_path = os.path.join(results_dir, 'gaussian_dataset.pt')
    initial_params_path = os.path.join(results_dir, 'initial_params.pt')
    final_params_path = os.path.join(results_dir, 'final_params.pt')
    loss_history_path = os.path.join(results_dir, 'loss_history.pt')
    initial_graph_path = os.path.join(results_dir, 'initial_states.png')
    converged_graph_path = os.path.join(results_dir, 'converged_states.png')

    # Check if we should load saved results
    # List to store trained models, initial and final parameters, and loss history
    initial_weights = torch.zeros((num_models, 2), device=device)
    initial_biases = torch.zeros((num_models,), device=device)
    final_weights = torch.zeros((num_models, 2), device=device)
    final_biases = torch.zeros((num_models,), device=device)
    loss_history = torch.zeros((num_models, num_epochs), device=device)

    if load_existing and os.path.exists(final_params_path):
        print(f"Loading existing data from {results_dir}")
        dataset_data = torch.load(dataset_path)
        initial_params = torch.load(initial_params_path)
        final_params = torch.load(final_params_path)
        loss_history = torch.load(loss_history_path)

        dataset = GaussianDataset(device=device, preloaded_data=dataset_data)
    else:
        # Generate Gaussian Dataset and Cache on GPU
        mean = torch.tensor([3.0, 2.0], device=device)
        cov = torch.tensor([[2.0, -1.0], [-1.0, 1.0]], device=device)

        print("Creating Gaussian data set")
        dataset = GaussianDataset(num_samples, mean, cov, device)

        # DataLoader
        print("Creating Gaussian data loader")
        batch_size = 1024
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # List to store trained models
        trained_models = []

        # Uniform sampling range for weights
        sample_range = (-8, 8)

        activation = get_activation(activation_name)

        for i in range(num_models):
            # Initialize model
            model = SimpleNN(input_dim=2, activation=activation).to(device)

            # mean_centered_initialization(model, length_range=(1., 1.))

            # uniform point selection
            # intersection = torch.FloatTensor(2).uniform_(sample_range[0], sample_range[1])
            # data sampling
            intersection = dataset.X[torch.randint(0, dataset.X.size(0), (1,)).item()]

            # # Initialize weights and biases
            init_bias_intersections(
                model.linear, 
                intersection)

            # make sure the node is not dead
            with torch.no_grad():
                while model(dataset.X).sum().item() == 0:
                    print("Reinitializing dead model")
                    # mean_centered_initialization(model)
                    init_bias_intersections(
                        model.linear, 
                        torch.FloatTensor(2).uniform_(sample_range[0], sample_range[1]))

            # Save initial parameters
            initial_weights[i] = model.linear.weight.clone().detach().view(2)
            initial_biases[i] = model.linear.bias.clone().detach().view(1)

            # Define loss and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
            scheduler = None # optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

            # Trainer
            trainer = Trainer(model, data_loader, criterion, optimizer, scheduler, device)

            losses = trainer.train(num_epochs)

            # Save loss curve
            loss_history[i] = losses
            # Save final parameters
            final_weights[i] = model.linear.weight.clone().detach().view(2)
            final_biases[i] = model.linear.bias.clone().detach().view(1)
            
            print(f"Model {i} Loss {losses[-1]}")
            # plot_epoch_losses(losses)

            # Store the trained model
            trained_models.append(model)

            if (i + 1) % 100 == 0:
                print(f'Trained {i + 1}/{num_models} models')

        print(f'Finished training {num_models} models with {activation_name} activation.')

        initial_params = { 'weights': initial_weights, 'biases': initial_biases}
        final_params = { 'weights': final_weights, 'biases': final_biases}

        # Save results to disk
        torch.save({'X': dataset.X, 'Y': dataset.Y, 'mean': dataset.mean, 'cov_inv': dataset.cov_inv}, dataset_path)
        torch.save(initial_params, initial_params_path)
        torch.save(final_params, final_params_path)
        torch.save(loss_history, loss_history_path)
        print(f"Results saved to {results_dir}")


    initial_means = project_onto_decision_boundaries(
        dataset.mean, 
        initial_params['weights'], 
        initial_params['biases'])
    initial_vecs = get_scaled_vectors(initial_params['weights'])

    final_means = project_onto_decision_boundaries(
        dataset.mean, 
        final_params['weights'], 
        final_params['biases'])
    final_vecs = get_scaled_vectors(final_params['weights'])

    final_losses = loss_history[:,-1]
    
    distances = hyperplane_distances(
        dataset.mean, 
        final_params['weights'], 
        final_params['biases'])
    # print(distances)
    classes = classify_metrics(distances, [1.0])
    classes[final_losses > 1.7] = 2
    
    # print(final_losses > 1.0)
    # print(classes)
    print(final_losses)

    # plot_ex1_results(dataset.X, linear_means, linear_vecs)
    plot_ex1_results(
        dataset.X, 
        initial_means, initial_vecs, classes,
        title=f"{activation_name} Initial Weights",
        xlabel="x0", ylabel="x1",
        filename=initial_graph_path)
    plot_ex1_results(
        dataset.X, 
        final_means, final_vecs, classes,
        title=f"{activation_name} Converged Weights",
        xlabel="x0", ylabel="x1",
        filename=converged_graph_path)

    # Initialize metrics
    total_error = loss_history[:, -1].mean().item()  # Final error (average across all models)
    coherent_error = loss_history[classes == 0, -1].mean().item() if (classes == 0).sum() > 0 else 0  # Error for "Coherent" class
    adhoc_error = loss_history[classes == 1, -1].mean().item() if (classes == 1).sum() > 0 else 0  # Error for "Adhoc" class

    # Class counts
    coherent_count = (classes == 0).sum().item()  # Number of models classified as "Coherent"
    adhoc_count = (classes == 1).sum().item()  # Number of models classified as "Adhoc"
    dead_count = (classes == 2).sum().item()  # Number of models classified as "Dead"

    # Print out the stats in table format
    print("\nSingle Linear Node Results:")
    print(f"{'Error':<20} {total_error:<10.3f}")
    print(f"{'Coherent Error':<20} {coherent_error:<10.3f}")
    print(f"{'Adhoc Error':<20} {adhoc_error:<10.3f}")
    print(f"{'Coherent Count':<20} {coherent_count:<10}")
    print(f"{'Adhoc Count':<20} {adhoc_count:<10}")
    print(f"{'Dead Count':<20} {dead_count:<10}")
