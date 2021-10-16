'''
this is the classifier to be used
first initialize with all the required data
then class predict
after that, call compute_accuracy
done
'''

import numpy as np

class knn_classifier:
    '''
        This knn classifier constructs a classification for all possible k
        May change later
    '''
    def __init__(self, X_train, X_test, y_train, y_test, distance_function):
        
        self.X_train = X_train            # converting to d dimensional matrices
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

        self.distance_function = distance_function
        pass

    def predict(self):
        '''
            Computes an array of predictions of X_test
        '''
        self.predictions = []
        idx2 = 0
        for new_point in self.X_test:
            neighbors = []
            curr_predictions = []
            idx = 0
            
            indices = np.arange(len(self.X_train))
            distances = np.array([self.distance_function(new_point, given_point) for given_point in self.X_train])

            neighbors = np.column_stack((distances, indices))
            
            neighbors = neighbors[neighbors[:,0].argsort()]     # sorting by distances

            spam_labels = 0
            ham_labels = 0

            # so here we will find the values by varying the number of neighbors from 1 to len(X_train)

            for k in range(len(neighbors)):
                index = int(neighbors[k][1])
                given_classification = self.y_train[index]
                if given_classification == 1:
                    spam_labels += 1
                else:
                    ham_labels += 1
                
                if k == 0:
                    continue

                if spam_labels > ham_labels:
                    curr_predictions.append(1)
                else:
                    curr_predictions.append(0)

            self.predictions.append(curr_predictions)

            if idx2 % 100 == 0:
                print(f"Done for {idx2 + 1} test samples")

            idx2 += 1

        return
    
    def compute_accuracy(self):
        '''
            This will have to compute accuracy
            for variable no of neighbors
            i.e. from 1 to len(X_train)
        '''
        accuracies = []
        num_nbrs = []

        for k in range(len(self.y_train) - 1):
            curr_predictions = np.array([predicted_value[k] for predicted_value in self.predictions])
            cnt = 0
            for i in range(len(self.y_test)):
                if self.y_test[i] == curr_predictions[i]:
                    cnt += 1
            accuracies.append(cnt / len(self.y_test))
            num_nbrs.append(k + 1)

        return accuracies, num_nbrs
