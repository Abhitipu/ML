import numpy as np
import matplotlib.pyplot as plt
from model import Network
from data_handling import my_dataset
from torch.utils.data import DataLoader
import torch

if __name__ == "__main__":
    # Load the data
    training_datapath = "../input_files/sat.trn"
    test_datapath = "../input_files/sat.tst"
    
    training_dataset = my_dataset(training_datapath)
    training_loader = DataLoader(training_dataset, batch_size=1, shuffle=True)
    # Can iterate using for labels, value in training_loader:

    test_dataset = my_dataset(test_datapath)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    # Set the device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define the hyperparameters
    input_size = 36
    output_size = 7
    
    # # Define the NN
    # for learning_rate in required_learning_rates:
    
    required_learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    required_hidden_layers = [[], [2], [6], [2, 3], [3, 2]]

    for hidden_layers in required_hidden_layers:
        curr_network = Network(input_size=input_size, output_size=output_size, hidden_layers=hidden_layers)
        print(curr_network)
        
    # n_epochs = 100
    # for epoch in range(n_epochs):