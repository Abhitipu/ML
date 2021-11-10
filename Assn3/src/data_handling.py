import torch
from torch.utils.data import Dataset
import numpy as np

class my_dataset(Dataset):
    def __init__(self, datapath, transform=None):
        matrix = np.loadtxt(datapath, dtype=float)
        # converting to a tensor
        data = torch.from_numpy(matrix)
        X = data[:,:-1]
        self.Y = data[:,-1]
        self.Y -= 1         # following 0 based indexing
        
        # we normalize the data in the beginning
        X_mean = X.mean(dim=0)
        X_var = X.var(dim=0)
        self.X = (X - X_mean) / X_var

    def __getitem__(self, index):
        assert(index < len(self))
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)
