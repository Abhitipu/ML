# All the distance and similarity functions will be put here
import numpy as np

def cosine_similarity(X, Y):
    X_mag = np.sqrt(sum([i**2 for i in X]))
    Y_mag = np.sqrt(sum([i**2 for i in Y]))
    dot_prod = sum([x*y for x, y in X, Y])
    return dot_prod / (X_mag * Y_mag)

def manhattan_distance(X, Y):
    ans = sum([np.abs(x - y) for x, y in X, Y])
    return ans

def euclidian_distance(X, Y):
    ans = np.sqrt(sum([(x - y)**2 for x, y in X, Y]))
    return ans
