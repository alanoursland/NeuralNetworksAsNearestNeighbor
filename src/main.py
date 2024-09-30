# main.py

import torch
import numpy as np
import random
from experiments.experiment1 import run_experiment1
from experiments.experiment2 import run_experiment2
from experiments.experiment3 import run_experiment3
from utils.utils import set_seed

def main():
    set_seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Experiment 1
    print("Running Experiment 1 with ReLU activation")
    run_experiment1(device, num_samples=3000, num_models=1000, num_epochs=100, activation_name='ReLU')
    
    print("Running Experiment 1 with Abs activation")
    run_experiment1(device, num_samples=3000, num_models=1000, num_epochs=100, activation_name='Abs')

    # Experiment 2
    print("Running Experiment 2 with ReLU activation")
    run_experiment2(device, activation_name='ReLU')
    
    print("Running Experiment 2 with Abs activation")
    run_experiment2(device, activation_name='Abs')

    # Experiment 3
    print("Running Experiment 3 with ReLU activation")
    run_experiment3(device, activation_name='ReLU')
    
    print("Running Experiment 3 with Abs activation")
    run_experiment3(device, activation_name='Abs')

if __name__ == "__main__":
    main()
