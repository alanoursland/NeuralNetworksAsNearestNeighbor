# e2_nn_stats.py

import torch
from experiments.experiment2 import load_experiment, calculate_experiment_stats, calculate_l2_norm_error_count, plot_experiment_curves

def main():
    exp_data = {
        'ReLU': load_experiment("results/ex2/ReLU"),
        'ReLU_sampled': load_experiment("results/ex2/ReLU_sampled"),
        'Abs': load_experiment("results/ex2/Abs"),
        'Abs_sampled': load_experiment("results/ex2/Abs_sampled"),
    }

    # Access data
    # print(exp_abs.keys())
    # print(exp_abs['001'].keys())
    # print(exp_abs['001']['experiment_info'])  # Get info for Abs experiment instance 001

    for instance_key, instance_data in exp_data.items():
        print(instance_key)
        print(calculate_experiment_stats(instance_data))
        print()

    for instance_key, instance_data in exp_data.items():
        print(instance_key)
        print(calculate_l2_norm_error_count(instance_data))

    plot_experiment_curves(exp_data)

if __name__ == "__main__":
    main()
