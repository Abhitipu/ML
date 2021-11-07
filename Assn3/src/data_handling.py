import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np

class my_dataset(DataLoader):
    def __init__(self, datapath, transform=None):
        matrix = np.loadtxt(datapath)
        # converting to a tensor
        data = torch.from_numpy(matrix)
        self.X = data[:,:-1]
        self.Y = data[:,-1]

    def __getitem__(self, index):
        assert(index < len(self))
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)
