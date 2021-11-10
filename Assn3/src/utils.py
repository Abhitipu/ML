import numpy as np

class PCA():   
    def __init__(self, X):           
        covariance_matrix = np.cov(X - X.mean(axis=0), rowvar=False)    
        eigen_values, eigen_vector = np.linalg.eigh(covariance_matrix)        
        self.U = np.asarray(eigen_vector).T[::-1]    
        self.D = eigen_values[::-1]

    def project(self, X, new_dim):
        Z = np.dot(X-X.mean(axis=0),np.asmatrix(self.U[:new_dim]).T)
        return Z
    