# e2_nn.py

import torch
from experiments.experiment2 import train_nn
from utils.utils import set_seed

def main():
    # set_seed(3)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    load_existing=False

    for _ in range(10):
        train_nn("ReLU", device, activation_name='ReLU', use_bias_sampling=False, load_existing=load_existing)
        train_nn("ReLU_sampled", device, activation_name='ReLU', use_bias_sampling=True, load_existing=load_existing)
        train_nn("Abs", device, activation_name='Abs', use_bias_sampling=False, load_existing=load_existing)
        train_nn("Abs_sampled", device, activation_name='Abs', use_bias_sampling=True, load_existing=load_existing)

if __name__ == "__main__":
    main()
