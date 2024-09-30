# experiments/experiment2.py

import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from activations.activations import get_activation
from data.mnist_data import get_mnist_dataset
from evaluation.evaluator import Evaluator
from models.feedforward_nn import FeedForwardNN
from models.kmeans_classifier import KMeansClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from training.trainer import Trainer


def get_next_experiment_dir(results_dir):
    # Ensure the base directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Start numbering from 001
    run_number = 1

    while True:
        # Format the new directory name using zero-padded numbering
        new_dir = os.path.join(results_dir, f'{run_number:03d}')
        
        # Check if the directory exists
        if not os.path.exists(new_dir):
            # Return the new directory path if it doesn't exist
            return new_dir
        
        # Increment the run number for the next check
        run_number += 1

def initialize_weights_he(model):
    for m in model.modules():
            if isinstance(m, nn.Linear):
                # He initialization for the weights
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                # Initialize the biases to zero
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

def initialize_biases_sampling(model, dataset, device='cpu'):
    """
    Initialize the biases of the first linear layer in the model by sampling from the dataset.
    For the first linear layer, set the bias as: b = -W * z, where W is the weight matrix,
    and z is a sample from the dataset.
    
    Args:
        model: The neural network model (assumed to have linear layers).
        dataset: The dataset from which we sample z (assumed to be MNIST, i.e., a set of images).
        device: The device (CPU/GPU) where the model and dataset are loaded.
    """
    # Stack the dataset into a tensor (e.g., for MNIST)
    data = torch.stack([data[0] for data in dataset]).to(device)  # Sample data (excluding labels)

    # Boolean to track if we have processed the first layer
    first_layer_initialized = False

    # Iterate through model layers
    for m in model.modules():
        if isinstance(m, nn.Linear) and not first_layer_initialized:
            # Sample a random input z from the dataset
            idx = torch.randint(0, len(data), (1,)).item()  # Random index
            z = data[idx].view(-1)  # Flatten the image (if dataset is MNIST)

            # Make sure z is on the same device as the model
            z = z.to(device)
            
            # Compute bias: b = -W * z
            if m.weight is not None:
                W = m.weight  # Get the weights (W)
                b = -torch.matmul(W, z)  # Compute b = -W * z
                
                # Assign this value to the bias term
                if m.bias is not None:
                    m.bias.data = b

            # Mark that the first layer's bias has been initialized
            first_layer_initialized = True
            break  # Stop after the first linear layer

def init_model(hidden_size, activation_name, use_bias_sampling, train_dataset, device):
    activation = get_activation(activation_name)
    model = FeedForwardNN(input_dim=28*28, hidden_dim=hidden_size, output_dim=10, activation=activation).to(device)

    # Initialize weights
    initialize_weights_he(model)
    if use_bias_sampling:
        initialize_biases_sampling(model, train_dataset, device=device)
    return model

def train_model(model, train_loader, criterion, learning_rate, epochs, evaluator, device):
    # Define loss and optimizer
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    scheduler = None

    # Trainer
    trainer = Trainer(model, train_loader, criterion, optimizer, scheduler, device)
    train_loss, test_accuracy = trainer.train(num_epochs=epochs, evaluator=evaluator)
    return train_loss, test_accuracy

def plot_training_loss(train_loss, test_accuracy, save_path=None, title_loss="Training Loss Over Epochs", 
                       title_accuracy="Test Accuracy Over Epochs", xlabel="Epoch", ylabel_loss="Loss", ylabel_accuracy="1 - Accuracy"):
    error = 1 - test_accuracy

    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot Training Loss
    ax1.plot(train_loss, label="Train Loss", color='b')
    ax1.set_title(title_loss)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel_loss)
    ax1.set_yscale('log')
    ax1.grid(True)
    ax1.legend()

    # Plot Test Accuracy
    ax2.plot(error, label="1 - Accuracy", color='g')
    ax2.set_title(title_accuracy)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel_accuracy)
    ax2.set_yscale('log')
    ax2.grid(True)
    ax2.legend()

    # Adjust layout to avoid overlap
    plt.tight_layout()

    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path)

    # Show the plots
    plt.show()

def train_nn(experiment_name, device, activation_name='ReLU', use_bias_sampling=False, load_existing=False):
    results_dir = f"results/ex2/{experiment_name}"
    results_dir = get_next_experiment_dir(results_dir)

    print(f"Running {activation_name}. bias sampling = {use_bias_sampling}")
    batch_size = 64
    learning_rate = 0.001
    epochs = 50
    hidden_size = 128

    # Load MNIST Dataset
    train_dataset = get_mnist_dataset(train=True)
    test_dataset = get_mnist_dataset(train=False)

    # DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Move datasets to GPU
    # (Assuming enough GPU memory)
    # train_data = torch.stack([data[0] for data in train_dataset]).to(device)
    # train_targets = torch.tensor([data[1] for data in train_dataset]).to(device)
    test_data = torch.stack([data[0] for data in test_dataset]).to(device)
    test_targets = torch.tensor([data[1] for data in test_dataset]).to(device)

    # Model 1: Feedforward NN with ReLU or Abs
    model = init_model(hidden_size, activation_name, use_bias_sampling, train_dataset, device)

    # criterion = nn.MSELoss()
    criterion = nn.CrossEntropyLoss()
    evaluator = Evaluator(model, test_data, test_targets, criterion, device)
    train_loss, test_accuracy = train_model(model, train_loader, criterion, learning_rate, epochs, evaluator, device)

    print(f"Results {activation_name}. bias sampling = {use_bias_sampling}")
    losses, output, targets = evaluator.evaluate()
    stats = evaluator.generate_stats()

    # Save Results
    print(f"Writing experiment to {results_dir}")
    os.makedirs(results_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(results_dir, "model.pt"))
    torch.save(train_loss, os.path.join(results_dir, "train_loss.pt"))
    torch.save(test_accuracy, os.path.join(results_dir, "test_accuracy.pt"))
    torch.save(losses, os.path.join(results_dir, "losses.pt"))
    torch.save(output, os.path.join(results_dir, "output.pt"))
    torch.save(targets, os.path.join(results_dir, "targets.pt"))
    with open(os.path.join(results_dir, "experiment_info.txt"), 'w') as f:
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Hidden Size: {hidden_size}\n")
        f.write(f'Evaluation Loss: {stats["average_loss"]:.4f}\n')
        f.write(f'Accuracy: {stats["accuracy"]:.4f}\n')
        f.write(f'Precision: {stats["precision"]:.4f}\n')
        f.write(f'Recall: {stats["recall"]:.4f}\n')
        f.write(f'F1 Score: {stats["f1"]:.4f}\n')
        f.write(f'Confusion Matrix:\n{stats["confusion_matrix"]}\n')

    # Evaluator
    print(f'Train Loss: {train_loss[-1]:.6f}')
    print(f'Evaluation Loss: {stats["average_loss"]:.6f}')
    print(f'Accuracy: {stats["accuracy"]:.6f}')
    print(f'Precision: {stats["precision"]:.4f}')
    print(f'Recall: {stats["recall"]:.4f}')
    print(f'F1 Score: {stats["f1"]:.4f}')
    print(f'Confusion Matrix:\n{stats["confusion_matrix"]}')

def load_experiment_instance(path):
    data = {}

    # Load .txt file as raw data (not interpreted)
    experiment_info_path = os.path.join(path, "experiment_info.txt")
    with open(experiment_info_path, 'r') as f:
        data['experiment_info'] = f.read()

    # Load PyTorch tensors
    data['losses'] = torch.load(os.path.join(path, "losses.pt"))
    data['model'] = torch.load(os.path.join(path, "model.pt"))
    data['output'] = torch.load(os.path.join(path, "output.pt"))
    data['targets'] = torch.load(os.path.join(path, "targets.pt"))
    data['test_accuracy'] = torch.load(os.path.join(path, "test_accuracy.pt"))
    data['train_loss'] = torch.load(os.path.join(path, "train_loss.pt"))

    return data

def load_experiment(experiment_path):
    instances = {}
    if os.path.isdir(experiment_path):
        # Load each experiment instance in this subdirectory
        for instance_name in os.listdir(experiment_path):
            instance_path = os.path.join(experiment_path, instance_name)
            
            if os.path.isdir(instance_path):
                # Load the experiment instance
                instances[instance_name] = load_experiment_instance(instance_path)
    
    return instances


def calculate_experiment_stats(experiment_data):
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    test_errors = []

    # Loop through instances and calculate stats
    for _, instance_data in experiment_data.items():
        outputs = instance_data['output']  # Model predictions
        targets = instance_data['targets']  # True labels
        losses = instance_data['losses']  # Test errors (losses)

        # Convert logits or probabilities to class predictions
        predicted_classes = torch.argmax(outputs, dim=1)
        true_classes = targets

        # Calculate Accuracy
        correct_predictions = torch.sum(predicted_classes == true_classes).item()
        total_predictions = true_classes.size(0)
        accuracy = correct_predictions / total_predictions
        accuracies.append(accuracy)

        true_classes_np = true_classes.cpu().numpy()
        predicted_classes_np = predicted_classes.cpu().numpy()
        # Calculate Precision, Recall, F1 Score (per class, then average)
        precision = precision_score(true_classes_np, predicted_classes_np, average='macro')
        recall = recall_score(true_classes_np, predicted_classes_np, average='macro')
        f1 = f1_score(true_classes_np, predicted_classes_np, average='macro')

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

        # Test Error (mean of losses)
        test_error = torch.mean(losses).item()
        test_errors.append(test_error)

    # Calculate unique errors and max error count
    unique_errors_count, consistent_errors_count, avg_error_count, voting_error_count = calculate_error_diversity(experiment_data)

    # Calculate overall stats
    stats = {
        'Accuracy (Avg)': np.mean(accuracies),
        'Accuracy (Min)': np.min(accuracies),
        'Accuracy (Max)': np.max(accuracies),
        'Precision': np.mean(precisions),
        'Recall': np.mean(recalls),
        'F1 Score': np.mean(f1_scores),
        'Test Error': np.mean(test_errors),
        'Unique Errors': unique_errors_count,
        'Avg Error Count': avg_error_count,
        'Consistent Error Count': consistent_errors_count,
        'Voting Error Count': voting_error_count
    }

    return stats

def calculate_error_diversity(experiment_data):
    """
    Calculate the number of unique errors and the error that appeared in the most models.
    
    Args:
        experiment_data (dict): A dictionary containing experiment instances as values.
        
    Returns:
        tuple: (unique_errors_count, max_error_count)
    """
    # Initialize an accumulation tensor for counting errors
    first_instance = next(iter(experiment_data.values()))
    total_errors = torch.zeros_like(first_instance['targets'])  # Same size as targets, filled with zeros

    num_models = len(experiment_data)

    # Loop through each instance and accumulate errors
    for _, instance_data in experiment_data.items():
        outputs = instance_data['output']
        targets = instance_data['targets']
        
        # Convert logits or probabilities to class predictions
        predicted_classes = torch.argmax(outputs, dim=1)
        
        # Create error tensor (1 for incorrect predictions, 0 for correct)
        error_tensor = (predicted_classes != targets).int()
        
        # Accumulate errors (add 1 where there is an error)
        total_errors += error_tensor

    # Count the number of unique errors (non-zero elements in the accumulated tensor)
    unique_errors_count = torch.count_nonzero(total_errors).item()

    # Find the maximum error count (the maximum value in the accumulated error tensor)
    consistent_errors_count = torch.sum(total_errors == num_models).item()
    voting_errors_count = torch.sum(total_errors >= num_models/2).item()

    # Calculate the average error count (mean of non-zero elements in the accumulated tensor)
    avg_error_count = torch.sum(total_errors).item() / num_models

    return unique_errors_count, consistent_errors_count, avg_error_count, voting_errors_count


def calculate_l2_norm_error_count(experiment_data):
    # Get the number of instances from the first model's outputs
    first_instance = next(iter(experiment_data.values()))
    output_shape = first_instance['output'].shape
    device = first_instance['output'].device
    num_instances = output_shape[0]
    num_features = output_shape[1]
    num_models = len(experiment_data)

    # Initialize tensor to store L2 norms across models
    l2_norms = torch.zeros(output_shape, device=device)

    # Compute L2 norm for each model's output
    for _, (_, instance_data) in enumerate(experiment_data.items()):
        l2_norms += torch.square(instance_data['output'])  

    # Find the model with the minimum L2 norm for each instance
    min_norm_indices = torch.argmin(l2_norms, dim=1)

    # Get the true labels (targets) from any instance (since targets are the same for all models)
    true_labels = first_instance['targets']

    errors = (min_norm_indices != true_labels).int()
    error_count = torch.sum(errors)

    print(f"Total Error Count based on L2 Norms: {error_count}")

    # plot_training_loss(train_loss, test_accuracy)
        
def plot_experiment_curves(exp_data):
    """
    Plots average loss and 1-accuracy curves across instances for each experiment.
    
    Args:
        exp_data (dict): Dictionary containing experiment data for each experiment.
    """

    # Initialize lists to hold data for plotting
    experiment_names = list(exp_data.keys())
    loss_curves = {name: [] for name in experiment_names}
    accuracy_curves = {name: [] for name in experiment_names}

    # Loop through each experiment and calculate average loss and 1-accuracy across instances
    for exp_name, instances in exp_data.items():
        all_loss_curves = []
        all_accuracy_curves = []

        for _, instance_data in instances.items():
            # Load training loss and test accuracy tensors
            train_loss = instance_data['train_loss']  # shape: [epochs]
            test_accuracy = instance_data['test_accuracy']  # shape: [epochs]
            
            all_loss_curves.append(train_loss.cpu().numpy())
            all_accuracy_curves.append(test_accuracy.cpu().numpy())
        
        # Calculate average loss and 1-accuracy across instances
        avg_loss = np.mean(all_loss_curves, axis=0)
        avg_1_minus_accuracy = 1 - np.mean(all_accuracy_curves, axis=0)

        loss_curves[exp_name] = avg_loss
        accuracy_curves[exp_name] = avg_1_minus_accuracy

    # Plot the average loss curves for each experiment
    plt.figure(figsize=(6, 4))
    for exp_name in experiment_names:
        plt.plot(loss_curves[exp_name], label=exp_name)
    plt.yscale('log')
    plt.xlim(left=0)
    plt.title("Average Loss Curves")
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig('results/ex2/ex2_training_error.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot the average 1-accuracy curves for each experiment
    plt.figure(figsize=(6, 4))
    for exp_name in experiment_names:
        plt.plot(accuracy_curves[exp_name], label=exp_name)
    plt.yscale('log')
    plt.xlim(left=0)
    plt.title("Average 1-Accuracy Curves")
    plt.xlabel("Epoch")
    plt.ylabel("1 - Average Test Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig('results/ex2/ex2_test_accuracy.png', dpi=300, bbox_inches='tight')
    plt.show()



# # Model 3: KMeans Classifier
# # Prepare data for KMeans
# num_clusters = 50  # Can be adjusted
# kmeans_model = KMeansClassifier(num_clusters=num_clusters, input_dim=28*28).to(device)
# kmeans_model.fit(train_data.view(-1, 28*28).cpu().numpy())

# # Train linear layer
# # Create a DataLoader for KMeans distances
# class KMeansDataset(torch.utils.data.Dataset):
#     def __init__(self, data, targets, kmeans_model):
#         self.data = data
#         self.targets = targets
#         self.kmeans_model = kmeans_model

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         x = self.data[idx].view(-1).cpu().numpy()
#         distances = self.kmeans_model.kmeans.transform([x])[0]
#         distances = torch.from_numpy(distances).float()
#         y = self.targets[idx]
#         return distances.to(device), y.to(device)

# kmeans_train_dataset = KMeansDataset(train_data, train_targets, kmeans_model)
# kmeans_train_loader = torch.utils.data.DataLoader(kmeans_train_dataset, batch_size=batch_size, shuffle=True)

# kmeans_test_dataset = KMeansDataset(test_data, test_targets, kmeans_model)
# kmeans_test_loader = torch.utils.data.DataLoader(kmeans_test_dataset, batch_size=batch_size, shuffle=False)

# # Define loss and optimizer for linear layer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(kmeans_model.linear.parameters(), lr=0.01)

# # Trainer
# trainer = Trainer(kmeans_model, kmeans_train_loader, criterion, optimizer, device)
# trainer.train(num_epochs=10)

# # Evaluator
# evaluator = Evaluator(kmeans_model, kmeans_test_loader, criterion, device)
# evaluator.evaluate()

