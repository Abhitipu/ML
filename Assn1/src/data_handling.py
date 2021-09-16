import pandas as pd
from description import attr_list, possible_values, classifications, header_list

class my_data:
    def __init__(self, input_file):
        self.df = pd.read_csv(input_file, header=None, names=header_list)

    def gen_test_and_validation_set(self):          # use this info! Not using this currently!
        self.df = self.df.sample(frac = 1).reset_index(drop=True)                    # randomize the data set
        self.training_set = self.df[:int(0.8*len(self.df))]
        self.validation_set = self.df[int(0.8*len(self.df)):]
    
    def get_classification(self, indices):
        '''
            This will just return the classification of a list of indices in a dictionary.
            Frequency of the class values in a list
        '''

        my_values = dict()
        for classification in classifications:
            my_values[classification] = 0

        for idx in indices:
            my_values[self.training_set['Class_value'][idx]] += 1

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
            myFreq[self.training_set[attribute][idx]].append(idx)
        
        return myFreq
