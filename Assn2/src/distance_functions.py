# All the distance and similarity functions will be put here
import numpy as np

# just verify this once!
def cosine_similarity(X, Y):
    '''
        Returns the cosine similaritiy between 2 vectors
        Here, 1e-35 is added to prevent division by zero
        We return 1 - cos(x) to get the distance between the two vectors
    '''
    cos_theta = np.dot(X, Y) / (np.linalg.norm(X) * np.linalg.norm(Y) + 1e-35)
    return 1 - cos_theta

def manhattan_distance(X, Y):
    '''
        Returns the manhattan distance between two vectors
    '''
    ans = np.abs(X - Y)
    return np.sum(ans)

def euclidian_distance(X, Y):
    '''
        Returns the euclidian distance between two vectors
    '''
    ans = np.square(X - Y)
    return np.sqrt(np.sum(ans))
