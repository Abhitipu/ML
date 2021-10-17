'''
    This file handles our input dataset.
    It is also reponsible for splits, preprocessing and vectorization
'''

import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class my_data:
    def __init__(self, input_file):
        '''
            Reads from an input csv file using pandas
        '''

        self.df = pd.read_csv(input_file)                                                           # reading csv
        self.messages = np.array([self.preprocess(message) for message in self.df["text"].values])  # numpy array of messages
        self.labels = np.array(self.df["label_num"].values)                                         # the classification labels

        self.vectorize()
        # Because our code is good with numbers


    def preprocess(self, my_message):
        '''
            Here we make a few changes in the email messages
            TODO: Maybe modify later
        '''
        my_message = my_message.replace('\r',' ')
        my_message = my_message.replace('\n',' ')
        return my_message
    
    def vectorize(self):
        '''
            Convert the text to a matrix form with floating point values using Tfidf vectorizer
            We ignore english stop words to improve accuracy
        '''
        self.vectorizer = TfidfVectorizer(stop_words = ('english'))                             # a tf idf vectorizer gives a better accuracy
        self.normalized_data = self.vectorizer.fit_transform(self.messages)                     # converting the words to a vector 
        self.normalized_data = self.normalized_data.toarray()                                   # obtaining a 2-d matrix 
        return 

    def gen_test_and_validation_set(self):          
        '''
            Function to randomize and split the data set
            Training set : 80%
            Validation_set : 20%
        '''

        # Randomization
        perms = np.random.RandomState(seed = int(time.time())).permutation(len(self.labels))
        temp_labels = np.array([self.labels[i] for i in perms])
        temp_permutation = np.array([self.normalized_data[i] for i in perms])

        self.normalized_data, self.labels = temp_permutation, temp_labels
        
        # Splitting
        self.training_set = self.normalized_data[:int(0.8*self.normalized_data.shape[0])]           
        self.training_set_labels = self.labels[:int(0.8*len(self.labels))]


        self.validation_set = self.normalized_data[int(0.8*self.normalized_data.shape[0]):]
        self.validation_set_labels = self.labels[int(0.8*len(self.labels)):]
        
        return
