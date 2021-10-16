'''
    This file handles our input dataset.
    It is also reponsible for splits, preprocessing and vectorization
'''

import pandas as pd
import numpy as np
from description import attr_list, header_list
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import shuffle

class my_data:
    def __init__(self, input_file):
        '''
            Reads from an input csv file using pandas
        '''

        self.df = pd.read_csv(input_file, header=None, names=header_list)                       # reading csv
        self.messages = np.array([self.preprocess(message) for message in self.df["text"].values])
        self.labels = self.df.loc[:,"label_num"].values

        self.vectorize()


    def preprocess(self, my_message):
        '''
            Make few changes in the email messages
        '''
        my_message = my_message.replace('\r',' ')
        my_message = my_message.replace('\n',' ')
        my_message = my_message.replace('#',' ')
        my_message = my_message.replace("we ' re","we are")
        my_message = my_message.replace("they ' re","they are")
        my_message = my_message.replace("you ' re","you are")
        my_message = my_message.replace("Subject:"," ")

        return my_message
    
    def vectorize(self):
        '''
            Convert the text to a matrix for with floating point values
        '''
        self.vectorizer = TfidfVectorizer(stop_words = 'english')                               # a tf idf vectorizer gives a better accuracy
        self.normalized_data = self.vectorizer.fit_transform(self.messages)                   # converting the words to a vector TODO: need to verify!

        return 

    def gen_test_and_validation_set(self):          
        '''
            Function to randomize and split the data set
            Training set : 80%
            Validation_set : 20%
            It then normalizes the data using the tf idf vectorizer
        '''
        self.normalized_data, self.labels = shuffle(self.normalized_data, self.labels)                                    # randomize the data set
        
        self.training_set = self.normalized_data[:int(0.8*self.normalized_data.shape[0])]           # make the splits
        self.training_set_labels = self.labels[:int(0.8*len(self.labels))]

        self.validation_set = self.normalized_data[int(0.8*self.normalized_data.shape[0]):]
        self.validation_set_labels = self.labels[int(0.8*len(self.labels)):]

        return
