'''
this is the classifier to be used
first initialize with all the required data
then class predict
after that, call compute_accuracy
done
'''

import numpy as np

class knn_classifier:
    # need a dataset, distance function
    # need to make a generic distance function
    def __init__(self, X_train, X_test, y_train, y_test, distance_function, num_nbrs = 25):
        
        self.X_train = X_train.toarray()
        self.X_test = X_test.toarray()
        self.y_train = y_train
        self.y_test = y_test

        self.num_nbrs = num_nbrs
        self.distance_function = distance_function
        pass

    def predict(self):
        '''
            Computes an array of predictions of X_test
        '''
        print("Trying to predict")

        self.predictions = []
        idx2 = 0
        for new_point in self.X_test:
            neighbors = []
            idx = 0
            
            indices = np.arange(len(self.X_train))
            distances = np.array([self.distance_function(new_point, given_point) for given_point in self.X_train])

            neighbors = np.column_stack((distances, indices))
            
            neighbors = neighbors[neighbors[:,0].argsort()]     # sorting by distances

            k_nearest_neighbours = np.array([ int(neighbors[i][1]) for i in range(min(self.num_nbrs, len(neighbors))) ])
            # indices of the k nearest neighbours

            given_classification = np.array([self.y_train[i] for i in k_nearest_neighbours])
            # obtain the given classification

            class_values, counts = np.unique(given_classification, return_counts = True)
            assert(len(class_values) < 3)
            # get the corresponding classifications with the frequencies
            if idx2 == 0:
                print(class_values)
                print(counts)

            max_idx = np.argmax(counts)
            # get the index for the maxima

            if idx2 == 0:
                print(max_idx)
                print(class_values[max_idx])

            self.predictions.append(class_values[max_idx])
            # append the prediction thus obtained
            
            if idx2 % 100 == 0:
                print(f"Done for {idx2}")

            idx2 += 1

        return
    
    def compute_accuracy(self):
        cnt = 0
        for i in range(len(self.y_test)):
            if self.y_test[i] == self.predictions[i]:
                cnt += 1

        return cnt / len(self.y_test)
