import torch
from torch import nn

# A generic neural network model with an arbitrary number of hidden layers
class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        '''Initializes the network with given params and random weights'''
        super(Network, self).__init__()
        
        if len(hidden_layers) == 0:
            self.layers = nn.ModuleList([nn.Linear(input_size, output_size)])
        else:
            self.layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
            self.layers.extend([nn.Linear(hidden_layers[i], hidden_layers[i+1]) for i in range(len(hidden_layers)-1)])
            self.layers.extend([nn.Linear(hidden_layers[-1], output_size)])
            
    def forward(self, x):
        '''The feed forward method'''
        for i in range(len(self.layers)-1):
            x = self.layers[i](x)
            x = torch.relu(x)
        return self.layers[-1](x)