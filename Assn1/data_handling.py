import pandas as pd
from description import attr_list, possible_values, classifications

class my_data:
    def __init__(self, input_file):
        self.df = pd.read_csv(input_file, header=None, names=attr_list)
        
    
    def get_classification(self, indices):
        '''
            This will just return the classification of a list of indices in a dictionary.
        '''

        my_values = Dict()
        for classification in classifications:
            my_values[classification] = 0

        for idx in indices:
            my_values[self.df[idx][class_value]] += 1

        return my_values
            
    def split_values(self, attribute, indices):
        '''
            This will split the values by the attribute,
            Then it will return a dictionary of the list of indices supported by the new conjunction
        '''

        myFreq = Dict()    
        
        for value in possible_values[attribute]:
            myFreq[value] = []
        
        for idx in indices:
            myFreq[df[idx][attribute]].append(idx)
        
        return myFreq
