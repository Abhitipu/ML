import torch
from torch.utils.data import Dataset
import numpy as np

'''Class that stores the dataset'''
class my_dataset(Dataset):
    def __init__(self, datapath, transform=None):
        '''Normalizes the data before storing'''
        matrix = np.loadtxt(datapath, dtype=float)
        # converting to a tensor
        # data = torch.from_numpy(matrix)
        
        X = matrix[:,:-1]
        self.Y = matrix[:,-1].astype(int)
        self.Y -= 1         # following 0 based indexing
        
        # we normalize the data in the beginning
        X_mean = X.mean(0)
        X_var = X.var(0)
        self.X = (X - X_mean) / X_var

    '''Some magic methods to aid our design'''
    def __getitem__(self, index):
        assert(index < len(self))
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.X)
