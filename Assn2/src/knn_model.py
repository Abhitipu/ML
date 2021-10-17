'''
This is the classifier model. 
First we initialize with the dataset.
Then we find the nearest neighbors
After that we make predictions by varying k (no of neighbors)
'''

import sys
import numpy as np
from sklearn.metrics import classification_report

class knn_classifier:
    '''
        This knn classifier constructs a classification for all possible k
    '''
    def __init__(self, X_train, X_test, y_train, y_test, distance_function):
        '''
            First we initialize with the dataset and update the distance function
        '''
        
        self.X_train = X_train            
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.distance_function = distance_function
        pass

    def predict(self):
        '''
            Here, we compute a 2-d array of predictions for all instances in X_test
            Then we vary the number of neighbors and predict the class value
            This leads to a better time complexity
        '''
        self.predictions = []           # 2d array of predictions

        idx2 = 0
        for new_point in self.X_test:
            neighbors = []
            curr_predictions = []
            idx = 0
            
            # Obtain the indices and the distances from new_point
            indices = np.arange(len(self.X_train))
            distances = np.array([self.distance_function(new_point, given_point) for given_point in self.X_train])
            
            neighbors = np.column_stack((distances, indices))
            neighbors = neighbors[neighbors[:,0].argsort()]     # sorting by distances

            numerator = 0
            denominator = 1e-30

            # Here we will find the values by varying the number of neighbors from 1 to len(X_train)

            for k in range(len(neighbors)):
                index = int(neighbors[k][1])
                given_classification = self.y_train[index]
                
                curr_weight = 1 / (neighbors[k][0] + 1e-15)**2
                denominator += curr_weight

                if given_classification == 1:
                    numerator += curr_weight    # We add to the numerator iff it is classified as spam
                
                # We just check the weighted average
                # and choose the closer value
                if numerator / denominator > 0.5:
                    curr_predictions.append(1)
                else:
                    curr_predictions.append(0)

            self.predictions.append(curr_predictions)

            if idx2 % 100 == 0:                             # Printing status after processing every 100 test samples
                print(f"Done for {idx2 + 1} test samples", file=sys.stderr)

            idx2 += 1

        return
    
    def compute_accuracy(self):
        '''
            Here we compute accuracies by varying no of neighbors from 1 to len(X_train)
            After that we return a list containing the accuracies and the corresponding number of neighbors considered
        '''
        accuracies = []
        num_nbrs = []
        best_accuracy = 0
        best_num_nbrs = 0

        for k in range(len(self.y_train)):
            # Get all the predictions for the given k
            curr_predictions = np.array([predicted_value[k] for predicted_value in self.predictions])

            # Obtain accuracy
            cnt = 0
            for i in range(len(self.y_test)):
                if self.y_test[i] == curr_predictions[i]:
                    cnt += 1

            # Append to the accuracy list for plotting
            curr_accuracy = cnt / len(self.y_test)

            if curr_accuracy > best_accuracy:
                best_accuracy = curr_accuracy
                best_num_nbrs = k + 1

            accuracies.append(curr_accuracy)
            num_nbrs.append(k + 1)

        print(f"Obtained best accuracy {best_accuracy} for {best_num_nbrs} neighbors")
        best_predictions = np.array([predicted_value[k] for predicted_value in self.predictions])
        print(classification_report(self.y_test, best_predictions, target_names = ['Ham', 'Spam']))
        
        return accuracies, num_nbrs
