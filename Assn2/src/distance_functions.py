# All the distance and similarity functions will be put here
import numpy as np

def cosine_similarity(X, Y):
    if(np.linalg.norm(X) == 0):
        print("Nooooooooo")
    if(np.linalg.norm(Y) == 0):
        print("Nooooooooo")
    unit_X = X / np.linalg.norm(X)
    unit_Y = Y / np.linalg.norm(Y)
    return np.dot(unit_X, unit_Y)

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
