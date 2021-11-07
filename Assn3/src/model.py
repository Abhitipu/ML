# imports
import torch
import torch.nn as nn
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        ''' Builds a feedforward network with arbitrary hidden layers.

            Arguments
            ---------
            input_size: integer, size of the input layer
            output_size: integer, size of the output layer
            hidden_layers: list of integers, the sizes of the hidden layers
        '''
        super(Network, self).__init__()

        # Append the output size also to the hidden layers
        hidden_layers.append(output_size)

        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        # We make a list of tuples of consecutive layers.
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])

        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])

    def forward(self, x):
        ''' Forward pass through the network, returns the output logits '''
        # First we apply the linear function and then we perform the activtion function
        # Relu is the activation function where relu(x) = max(0, x)
        for layer in self.hidden_layers:
            x = F.relu(layer(x))          
        return x