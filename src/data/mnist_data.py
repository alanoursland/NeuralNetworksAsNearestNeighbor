# data/mnist_data.py

import torch
from torchvision import datasets, transforms

def get_mnist_dataset(train=True):
    transform = transforms.Compose(
        [transforms.ToTensor(), 
         transforms.Normalize((0.1307,), (0.3081,))])
    dataset = datasets.MNIST(root='./data', train=train, download=True, transform=transform)
    return dataset
