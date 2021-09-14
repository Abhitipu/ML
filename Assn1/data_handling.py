import pandas as pd
from description import attr_list, possible_values, classifications, header_list

class my_data:
    def __init__(self, input_file):
        self.df = pd.read_csv(input_file, header=None, names=header_list)
    
    def get_classification(self, indices):
        '''
            This will just return the classification of a list of indices in a dictionary.
        '''

        my_values = dict()
        for classification in classifications:
            my_values[classification] = 0

        for idx in indices:
            my_values[self.df['Class_value'][idx]] += 1

        return my_values
            
    def split_values(self, attribute, indices):
        '''
            This will split the values by the attribute,
            Then it will return a dictionary of the list of indices supported by the new conjunction
        '''

        myFreq = dict()    
        
        for value in possible_values[attribute]:
            myFreq[value] = []
        
        for idx in indices:
            myFreq[self.df[attribute][idx]].append(idx)
        
        return myFreq
