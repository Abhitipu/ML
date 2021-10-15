import numpy as np

# this is the classifier to be used
# first initialize with all the required data
# then class predict
# after that, call compute_accuracy
# done

class knn_classifier:
    # need a dataset, distance function
    # need to make a generic distance function
    def __init__(X_train, X_test, y_train, y_test, num_nbrs = 1000, distance_function):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.num_nbrs = num_nbrs
        self.distance_function = distance_function
        pass

    def predict():
    '''
        Might return an array of predictions of X_test
    '''
        self.predictions = []
        for new_point in X_test:
            neighbors = []
            for given_point in X_train:
                dist_val = self.distance_function(new_point, given_point)
                neighbors.append((dist_val, k))     # storing the distance with the index

            neighbors.sort() 
            # sort by distances

            k_nearest_neighbours = [ neighbors[i][1] for i in range(min(num_nbrs, X_train.shape[0])) ]
            # indices of the k nearest neighbours

            given_classification = Y_train[k_nearest_neighbours]
            # obtain the given classification

            class_values, counts = np.unique(given_classification, return_counts = True)
            # get the corresponding classifications with the frequencies

            max_idx = np.argmax(counts)
            # get the index for the maxima

            self.predictions.append[class_values[max_idx]]
            # append the prediction thus obtained

        return
    
    def compute_accuracy():
        cnt = 0
        for actual, predicted in self.y_test, self.predictions:
            if actual != predicted:
                cnt += 1

        return cnt / len(self.y_test)
