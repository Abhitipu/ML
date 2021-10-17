'''
This is the classifier model. 
First we initialize with the dataset.
Then we find the nearest neighbors
After that we make predictions by varying k (no of neighbors)
'''

import numpy as np

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

            spam_labels = 0
            ham_labels = 0

            # Here we will find the values by varying the number of neighbors from 1 to len(X_train)

            for k in range(len(neighbors)):
                index = int(neighbors[k][1])
                given_classification = self.y_train[index]
                if given_classification == 1:
                    spam_labels += 1
                else:
                    ham_labels += 1
                
                if k == 0:
                    continue

                # We just check the greater frequency
                # and classify accordingly
                if spam_labels > ham_labels:
                    curr_predictions.append(1)
                else:
                    curr_predictions.append(0)

            self.predictions.append(curr_predictions)

            if idx2 % 100 == 0:                             # Printing status after processing every 100 test samples
                print(f"Done for {idx2 + 1} test samples")

            idx2 += 1

        return
    
    def compute_accuracy(self):
        '''
            Here we compute accuracies by varying no of neighbors from 1 to len(X_train)
            After that we return a list containing the accuracies and the corresponding number of neighbors considered
        '''
        accuracies = []
        num_nbrs = []

        for k in range(len(self.y_train) - 1):
            # Get all the predictions for the given k
            curr_predictions = np.array([predicted_value[k] for predicted_value in self.predictions])

            # Obtain accuracy
            cnt = 0
            for i in range(len(self.y_test)):
                if self.y_test[i] == curr_predictions[i]:
                    cnt += 1

            # Append to the accuracy list for plotting
            accuracies.append(cnt / len(self.y_test))
            num_nbrs.append(k + 1)

        return accuracies, num_nbrs
