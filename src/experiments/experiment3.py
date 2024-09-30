# experiments/experiment3.py

import torch
from data.mnist_data import get_mnist_dataset
from models.lenet import LeNet
from activations.activations import get_activation
from training.trainer import Trainer
from evaluation.evaluator import Evaluator
from utils.utils import initialize_weights
import torch.nn as nn
import torch.optim as optim

def run_experiment3(device, activation_name='ReLU'):
    # Load MNIST Dataset
    train_dataset = get_mnist_dataset(train=True)
    test_dataset = get_mnist_dataset(train=False)

    # DataLoaders
    batch_size = 64
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Move datasets to GPU (if memory allows)
    # [Same as in Experiment 2]

    # Initialize activation
    activation = get_activation(activation_name)

    # Initialize model
    model = LeNet(activation).to(device)

    # Initialize weights
    initialize_weights(model)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # Trainer
    trainer = Trainer(model, train_loader, criterion, optimizer, device)
    trainer.train(num_epochs=10)

    # Evaluator
    evaluator = Evaluator(model, test_loader, criterion, device)
    evaluator.evaluate()
