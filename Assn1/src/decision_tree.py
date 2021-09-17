from description import attr_list, possible_values, classifications

class node:

    def __init__(self, conjunction, indices, impurity_evaluator, input_data):

        '''
            The constructor of a node in the decision tree
            The decision tree construction is done using the ID3 algorithm
            where we recursively decide on the attribute for making a
            split based on the degree to which they reduce randomization in the input.
        '''
    
        self.conjunction = conjunction                  # [dict: {attr_name -> attr_value}]: values of the attributes from the root to the node 
        self.indices = indices                          # [list]: indices of the entries in the dataFrame [corresponding to the conjunction] 

        self.is_root = True                             # indicates whether current node is the root of the tree

        for key in self.conjunction:                    # if no attribute value is assigned till now, its a root
            if self.conjunction[key] != '':
                self.is_root = False

        self.impurity_evaluator = impurity_evaluator    # function for deciding the attributes for splits
        
        self.attr = '-'                                 # Attribute to be checked in the current node
        self.children = dict()                          # children cprresponding to the different values of self.attr
        self.total = len(indices)                       # total no of nodes in the subtree

        self.input_data = input_data                    # Our input dataset
        self.idx = 0                                    # Unique identifier for a node
        self.is_leaf = True                             # Stores whether a node is a leaf

        self.compute_values()                           # compute initial values
        self.propagate()                                # extend the tree
        
        
        if(self.is_root):                               # do a dfs in the tree only if the node is a root
            self.assign_index()                         # assigns unique indices to each node
    
    def compute_values(self):
        '''
            We are finding out the classifications of our input data
            Based on that we will assign our class value by taking a majority vote
        '''

        self.class_value = ''                           # Attribute value which is later decided using a majority vote
        self.categories = self.input_data.get_classification(self.indices)      # Gets output from the dataset

        self.best_frequency = 0
        for key, value in self.categories.items():
            if value >= self.best_frequency:            # Majority is taken as the class_value
                self.best_frequency = value
                self.class_value = key

    def propagate(self):

        '''
            Now we have to construct out decision tree further
            We can stop if we have a pure split i.e. all our data points are in 1 category
            Or we have exhausted all attributes.
            Else we simply recurse further
        '''
        
        if self.best_frequency == self.total:       # pure node
            return
        
        done = True
        for attribute in attr_list:                 # if we haven't exhausted all attributes
            if self.conjunction[attribute] == '':
                done = False

        if done:
            return
        
        self.is_leaf = False                        
        max_gain = 0.0
        best_attribute = ''
        base_impurity = self.impurity_evaluator(self.categories)        # compute the impurity

        for attribute in attr_list:
            if self.conjunction[attribute] != '':                       # attribute is already chosen
                continue
            else:
                children_freq = self.input_data.split_values(attribute,self.indices)    # a dictionary of the nodes in the corresponding subtrees of the node's children
                
                curr_gain = base_impurity                               

                for key in children_freq :
                    new_classification = self.input_data.get_classification(children_freq[key])                           # compute for children
                    curr_gain -= (len(children_freq[key])/self.total)*self.impurity_evaluator(new_classification)   # subtract the weighted mean
                    
                if curr_gain >= max_gain:                               # re assign if necessary
                    max_gain = curr_gain
                    best_attribute = attribute

        new_conjunction = self.conjunction
        my_children = self.input_data.split_values(best_attribute, self.indices)  # finally split by the best attribute

        self.attr = best_attribute                                      # Set the attribute for the node
        
        for value in possible_values[best_attribute]:                   
            new_conjunction = dict()                        # making a deep copy to avoid reference errors
            for key, val in self.conjunction.items():
                new_conjunction[key] = val

            new_conjunction[best_attribute] = value                    # assigning the value

            if(len(my_children[value])):                               # In case there are no values we dont create the node
                self.children[value] = node(new_conjunction, my_children[value], self.impurity_evaluator, self.input_data)  


    def assign_index(self, cur = 0):                        
        '''
            dfs function to assign unique indices to each node
        '''
        self.idx = cur
        self.max_index = cur
        for key, child in self.children.items():
           self.max_index = child.assign_index(self.max_index+1)

        return self.max_index                               # returns the max index in the subtree
    
    def __str__(self):                  
        '''
            Magic method that comes in handy for printing the node using graphviz
        '''
        if not self.is_leaf:
            return f"Id = {self.idx}\n Attribute = {self.attr}\n Size = {len(self.indices)}\n Prediction = {self.class_value}"
        else:
            return f"Id = {self.idx}\n Size = {len(self.indices)}\n Target Value = {self.class_value}"

    def predict_cal(self, curr_conjunction, curr_height, height):
        '''
            Predicts the output value of the decision tree for a given conjunction
            This also keeps track of the current depth in the dfs
            This tracking comes in handy when we need to find the best possible depth
            for out decision tree.
        '''
        if(self.is_leaf or curr_height == height):      # when we cant go any further
            return self.class_value
        
        reqd_attribute = self.attr                      # check the node attribute
        reqd_value = curr_conjunction[reqd_attribute]   # check the required branch

        if reqd_value in self.children:                 # recurse if possible
            return self.children[reqd_value].predict_cal( curr_conjunction, curr_height+1, height)
        else:
            return self.class_value

    def predict_value(self, X_data, height=10):
        '''
            This function wraps around the predict_cal function.
            Given a validation set, it iterates over the set
            and gets the predictions from the predict_cal function.
            It then returns the list of predictions
        '''
        preds = []                                      # predictions made by our decision tree
        for index, row in X_data.iterrows():
            test_data = dict()
            for attribute in attr_list:
                test_data[attribute] = row[attribute]

            val = self.predict_cal(test_data, 1, height)    # check our prediction
            preds.append(val)

        return preds                                    # return the predictions
    
    def get_locations(self, locations):
        '''
            This is another dfs function useful for pruning.
            locations is a dictionary which maps to the nodes using
            their unique indices.
            This helps in quickly pruning a particular node instead 
            of having to do a dfs everytime we need to prune a node.
        '''
        locations[self.idx] = self
        for key, child in self.children.items():
            child.get_locations(locations)

    def alter_prune(self):
        '''
            Function to alter between a leaf and a non leaf node
            Basically it will just set it to a leaf.
            This can be checked while printing the tree
        '''
        self.is_leaf = not self.is_leaf
        return
