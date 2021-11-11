import numpy as np

class PCA():   
    '''Class used to perform PCA.'''
    def __init__(self, X):       
        '''It learns the projection matrix using the training set here'''    
        covariance_matrix = np.cov(X - X.mean(axis=0), rowvar=False)    
        eigen_values, eigen_vector = np.linalg.eigh(covariance_matrix)        
        self.U = np.asarray(eigen_vector).T[::-1]    
        self.D = eigen_values[::-1]

    def project(self, X, new_dim):
        '''Then it projects to a new dimension using the projection matrix'''
        Z = np.dot(X-X.mean(axis=0),np.asmatrix(self.U[:new_dim]).T)
        return Z
    