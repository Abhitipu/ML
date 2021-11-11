import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import Network
from data_handling import my_dataset
from utils import PCA

# Increase this for better accuracy    
n_epochs = 20

def train_network(curr_network, learning_rate, training_loader):
    '''Function to train the neural network using SGD and CrossEntropyLoss'''
    optimizer = torch.optim.SGD(curr_network.parameters(), lr=learning_rate)
    loss_function = torch.nn.CrossEntropyLoss()
    
    # Training the network
    for _ in tqdm(range(n_epochs)):
        curr_loss = 0.0
        for x, y in training_loader:
            predictions = curr_network(x.float())
            loss = loss_function(predictions, y.long())
            
            curr_loss += loss.item()
        
            # Back propagation
            optimizer.zero_grad()
            loss.backward()
            
            # update
            optimizer.step()
        # print(f"Epoch {_}: {curr_loss}")

def compute_accuracy(curr_network, test_loader):
    '''Function to compute accuracy of the neural network'''
    # Computation of accuracy
    curr_network.eval()
    n_correct = 0
    n_samples = 0
    # skip the gradient calculation while evaluation
    with torch.no_grad():
        # Here we iterate over the test_loader (containing mini batches)
        for x, y in test_loader:
            # Forward pass
            scores = curr_network(x.float())
            _, preds = scores.max(1) # preds is the index here
            
            n_correct += (preds == y).sum().item()
            n_samples += 1
                    
    return n_correct / n_samples

def compute_for_all_networks_and_plot(input_size, output_size, required_hidden_layers, required_learning_rates, training_loader, test_loader):
    ''' Q2 and 3 solver: Trains and computes accuracy for all networks Then constructs the required plots.'''
    best_accuracy = 0
    best_learning_rate = -1
    best_hidden_layers = []
    all_accuracies = []
    all_labels = []
    
    for hidden_layers in required_hidden_layers:
        accuracy_values = []
        for learning_rate in required_learning_rates:         
            curr_network = Network(input_size, output_size, hidden_layers)
            
            train_network(curr_network, learning_rate, training_loader)
            curr_accuracy =  compute_accuracy(curr_network, test_loader)
            accuracy_values.append(curr_accuracy)
                              
            if curr_accuracy > best_accuracy:
                best_accuracy = curr_accuracy
                best_learning_rate = learning_rate
                best_hidden_layers = hidden_layers
                
            print(f"Got accuracy {curr_accuracy} for {learning_rate} and {hidden_layers}")
        
        all_accuracies.append(accuracy_values)
        all_labels.append(f"L: " + ", ".join(str(x) for x in hidden_layers))
        
    print(f"Got best accuracy {best_accuracy} for {best_learning_rate} and {best_hidden_layers}")
    
    # Plot 1
    plt.title("Learning rates v/s accuracy for different models")
    plt.xlabel("Learning rates")
    plt.ylabel("Accuracy")
    plt.xscale("log")
    for label, accuracy_values in zip(all_labels, all_accuracies):
        plt.plot(required_learning_rates, accuracy_values, label=label)
    
    plt.legend()
    plt.savefig("../output_files/accuracy_vs_learning_rate.png")
    plt.cla()
    
    # Plot 2
    plt.title("Accuracy v/s model for different learning rates")
    plt.xlabel("Models")
    plt.ylabel("Accuracy")
    all_accuracies = np.array(all_accuracies)
    for i in range(len(required_learning_rates)):
        plt.plot(all_labels, all_accuracies[:,i], label=required_learning_rates[i]) 
    
    plt.legend()
    plt.savefig("../output_files/accuracy_vs_model.png")
    plt.cla()
    return best_accuracy, best_hidden_layers, best_learning_rate

def pca_n_scatterplot(training_dataset, output_size):
    '''Performs a dim reduction and makes a 2-d scatterplot'''
    pca = PCA(training_dataset.X)
    X_reduced = pca.project(training_dataset.X, 2)

    plt.title("Scatter plot after applying PCA")

    color_map =  np.array(["red", "green", "yellow", "blue", "orange", "purple", "brown"])	
    soil_map = ["red soil", "cotton crop", "grey soil", "damp grey soil", "soil with vegetation stubble", "mixture", "very damp grey soil"] 
    
    all_x_vals = []
    all_y_vals = []
    for i in range(output_size):
        x_vals = [] 
        y_vals = []
        for idx in range(len(training_dataset.Y)):
            if training_dataset.Y[idx] == i:
                x_vals.append(X_reduced.item(idx, 0))
                y_vals.append(X_reduced.item(idx, 1))
        all_x_vals.append(x_vals)
        all_y_vals.append(y_vals)
    
    for i in range(len(all_x_vals)):
        plt.scatter(all_x_vals[i], all_y_vals[i], c=color_map[i], label=soil_map[i]) 
    
    plt.legend()
    plt.savefig("../output_files/scatter_plot.png")
    plt.cla()

def learn_with_reduction(training_dataset, test_dataset, reduced_input_size, required_hidden_layers, best_learning_rate):
    '''Applies the algorithm on the reduced dimensions'''
    pca = PCA(training_dataset.X)
    X_reduced = pca.project(training_dataset.X, reduced_input_size)
    X_test_reduced = pca.project(test_dataset.X, reduced_input_size)
    
    # Regenerate the data loaders for the reduced dimensions
    training_dataset.Norm(X_reduced)
    training_loader = DataLoader(dataset=training_dataset, batch_size=1, shuffle=True)
    test_dataset.Norm(X_test_reduced)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)
    
    for hidden_layers in required_hidden_layers:
        curr_network = Network(reduced_input_size, output_size, hidden_layers)
        train_network(curr_network, best_learning_rate, training_loader)
        new_accuracy =  compute_accuracy(curr_network, test_loader)

        print(f"Got accuracy {new_accuracy} in the reduced dimension for {best_learning_rate} and {hidden_layers}")

if __name__ == "__main__":
    # Load the data
    training_datapath = "../input_files/sat.trn"
    test_datapath = "../input_files/sat.tst"
    
    training_dataset = my_dataset(training_datapath)
    training_loader = DataLoader(dataset=training_dataset, batch_size=1, shuffle=True)
    
    test_dataset = my_dataset(test_datapath)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

    # Define the hyperparameters
    input_size = 36
    output_size = 7
    
    required_learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    required_hidden_layers = [[], [2], [6], [2, 3], [3, 2]]
    
    # Q2, Q3
    best_accuracy, best_hidden_layers, best_learning_rate = compute_for_all_networks_and_plot(input_size, output_size, required_hidden_layers, required_learning_rates, training_loader, test_loader)
    # Q5
    pca_n_scatterplot(training_dataset, output_size)
    # Q6
    learn_with_reduction(training_dataset, test_dataset, 2, required_hidden_layers, best_learning_rate)