# e1.py

import torch
from experiments.experiment1 import run_experiment1
from utils.utils import set_seed

def main():
    set_seed(3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    load_existing=True
    # activation_name='Abs'
    activation_name='ReLU'

    run_experiment1(device, num_samples=1024, num_models=300, num_epochs=200, activation_name=activation_name, load_existing=load_existing)

if __name__ == "__main__":
    main()
