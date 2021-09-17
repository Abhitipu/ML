'''
    This file handles our input dataset.
'''

import pandas as pd
from description import attr_list, possible_values, classifications, header_list

class my_data:
    def __init__(self, input_file):
        '''
            Reads from an input csv file using pandas
        '''
        self.df = pd.read_csv(input_file, header=None, names=header_list)

    def gen_test_and_validation_set(self):          
        '''
            Function to randomize and split the data set
            Training set : 80%
            Validation_set : 20%
        '''
        self.df = self.df.sample(frac = 1).reset_index(drop=True)                    # randomize the data set
        self.training_set = self.df[:int(0.8*len(self.df))]                          # make the splits
        self.validation_set = self.df[int(0.8*len(self.df)):]
    
    def get_classification(self, indices):
        '''
            Function to just return the target values of a list of indices
            This helps us to know which target value is the majority for a node
        '''

        my_values = dict()
        for classification in classifications:
            my_values[classification] = 0

        for idx in indices:
            my_values[self.training_set['Class_value'][idx]] += 1       # increase the frequency by 1

        return my_values
            
    def split_values(self, attribute, indices):
        '''
            Function to split the values by the attribute,
            Returns a dictionary of the list of indices supported by the new conjunction
        '''

        myFreq = dict()    # frequency of the attribute values of a node
        
        for value in possible_values[attribute]:
            myFreq[value] = []
        
        for idx in indices:                                     # append the indices to the right branch
            myFreq[self.training_set[attribute][idx]].append(idx)
        
        return myFreq
