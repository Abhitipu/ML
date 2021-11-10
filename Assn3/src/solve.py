import matplotlib.pyplot as plt
from model import Network
from data_handling import my_dataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from utils import PCA

def add_line(x_label, y_label, learning_rate, hidden_layers):
    plt.title("Learning rates v/s accuracy for different networks")
    plt.xlabel("Learning rates")
    plt.ylabel("Accuracy")
    label = f"L: {learning_rate}" + ", ".join(str(x) for x in hidden_layers)
    plt.plot(x_label, y_label, label=label)

def save_n_clear(path):
    plt.savefig(path)
    plt.cla()
    
if __name__ == "__main__":
    # Load the data
    training_datapath = "../input_files/sat.trn"
    test_datapath = "../input_files/sat.tst"
    
    training_dataset = my_dataset(training_datapath)
    training_loader = DataLoader(dataset=training_dataset, batch_size=1, shuffle=True)
    # Can iterate using for labels, value in training_loader:
    
    print(len(training_dataset))
    
    test_dataset = my_dataset(test_datapath)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True)

    print(len(test_dataset))
    
    # Define the hyperparameters
    input_size = 36
    output_size = 7
    
    # required_learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    
    required_hidden_layers = [[], [2], [6], [2, 3], [3, 2]]
    required_learning_rates = [0.001]
    # required_hidden_layers = [[]]
    loss_function = torch.nn.CrossEntropyLoss()

    best_accuracy = 0
    best_learning_rate = -1
    best_hidden_layers = []
    
    for hidden_layers in required_hidden_layers:
        accuracy_values = []
        for learning_rate in required_learning_rates:         
            curr_network = Network(input_size, output_size, hidden_layers)
            n_epochs = 20
            optimizer = torch.optim.SGD(curr_network.parameters(), lr=learning_rate)

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
                            
            curr_accuracy =  n_correct / n_samples
            accuracy_values.append(curr_accuracy)
            
            if curr_accuracy > best_accuracy:
                best_accuracy = curr_accuracy
                best_learning_rate = learning_rate
                best_hidden_layers = hidden_layers
                
            print(f"Got accuracy {curr_accuracy} for {learning_rate} and {hidden_layers}")
            add_line(required_learning_rates, accuracy_values, learning_rate, hidden_layers)
        
    print(f"Got best accuracy {curr_accuracy} for {learning_rate} and {hidden_layers}")
    save_n_clear("../output_files/accuracy_plot.png")

    # plt.plot(x2, y2, label = "line 2")
    
    # pca = PCA(training_dataset.X)
    # X_reduced = pca.project(training_dataset.X, 2)
    # total_values = zip(X_reduced, training_dataset.y)
    
    # plt.title("Scatter plot after applying PCA")
    # # make a list of 7 colors
    # colors =  ["red", "green", "yellow", "blue", "orange", "purple", "brown"]	
    
    # plt.plot(total_values[:, 0], total_values[:, 1], color=colors[total_values[:,2].astype(int)])
        