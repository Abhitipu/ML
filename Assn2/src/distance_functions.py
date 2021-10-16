# All the distance and similarity functions will be put here
import numpy as np

# check this!
def cosine_similarity(X, Y):
    unit_X = X
    if np.linalg.norm(X) > 1e-50:
        unit_X /= np.linalg.norm(X)
    
    unit_Y = Y
    if np.linalg.norm(Y) > 1e-50:
        unit_Y /= np.linalg.norm(Y)

    return 1 - np.dot(unit_X, unit_Y)

def manhattan_distance(X, Y):
    ans = np.abs(X - Y)
    return np.sum(ans)

def euclidian_distance(X, Y):
    ans = np.square(X - Y)
    return np.sqrt(np.sum(ans))

if __name__ == "__main__":
    a = np.random.rand(10)
    b = np.random.rand(10)

    print(a)
    print(b)

    print(cosine_similarity(a, b))
    print(manhattan_distance(a, b))
    print(euclidian_distance(a, b))
