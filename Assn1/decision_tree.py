from description import attr_list, possible_values, classifications

class node:

    def __init__(self, conjunction, indices, impurity_evaluator, input_data):

        '''
            attr : each attribute is identified with an index (just a string maybe construct a hash)
            conjunction: values of the attributes from the root to the parent node(currently a list! convert to a 6 digit integer)
            categories: the classValues of the inputs [nCat1, nCat2, nCat3, nCat4]
            indices: indices of the entries in the dataFrame [corresponding to the conjunction]
            impurity_evaluator is the function which will evaluate our gains for splits
            2 possibilities: Gini index / Info gain
            idx : unique index of a node
        '''
    
        self.conjunction = conjunction                  # upto the parent
        self.indices = indices                          # a list of all the relevant indices

        self.is_root = True
        for key in self.conjunction:
            if self.conjunction[key] != '':
                self.is_root = False

        self.impurity_evaluator = impurity_evaluator    # for getting the best split in construction
        
        self.children = dict()                          # empty dict for now
        self.class_value = ''                           # useful for leaf nodes! Currently using most popular category only
        self.total = len(indices)                       # total no of nodes
        self.impurity = 0.0                             # need to compute impurity as well... is it required now?
        self.attr = '-'                                  # this will be decided later
        self.categories = dict()                        # contains the classifications of input
        self.input_data = input_data                    # This should rather be static!
        self.idx = 0

        self.compute_values()                           # compute initial values
        self.propagate()                                # extend the tree
        
        
        if(self.is_root):
            self.assign_index()
    
    def compute_values(self):
        '''
            We are finding out the classifications of our input data
        '''
        self.categories = self.input_data.get_classification(self.indices)

        self.best_frequency = 0
        for key, value in self.categories.items():
            if value >= self.best_frequency:
                self.best_frequency = value
                self.class_value = key

    def propagate(self):

        '''
            Now we have to construct it further
            We can stop if we have a pure split i.e. all out data points are in 1 category
            Or we have exhausted all attributes
        '''
        
        if self.best_frequency == self.total:       # pure node
            return
        
        self.done = True
        for attribute in attr_list:                 # if we haven't exhausted all attributes
            if self.conjunction[attribute] == '':
                self.done = False

        if self.done:
            return
        
        self.max_gain = 0.0
        self.best_attribute = ''
        self.base_impurity = self.impurity_evaluator(self.categories)        # advantage: both gini gain and info gain are calculated in the same manner

        for attribute in attr_list:
            if self.conjunction[attribute] != '':                       # attribute is already chosen
                continue
            else:
                self.children_freq = self.input_data.split_values(attribute,self.indices)    # a dictionary of the nodes in the corresponding subtree
                
                self.curr_gain = self.base_impurity                               

                for key in self.children_freq :
                    self.new_classification = self.input_data.get_classification(self.children_freq[key])
                    self.curr_gain -= (len(self.children_freq[key])/self.total)*self.impurity_evaluator(self.new_classification)   # subtract the weighted mean
                    
                if self.curr_gain >= self.max_gain:                               # re assign if necessary
                    self.max_gain = self.curr_gain
                    self.best_attribute = attribute

        self.new_conjunction = self.conjunction
        self.my_children = self.input_data.split_values(self.best_attribute, self.indices)

        self.attr = self.best_attribute                                      # Set the attribute for the node
        
        for value in possible_values[self.best_attribute]:                   # Stop if you have no values!
            new_conjunction = dict()
            for key, val in self.conjunction.items():
                new_conjunction[key] = val

            new_conjunction[self.best_attribute] = value

            if(len(self.my_children[value])):
                self.children[value] = node(new_conjunction, self.my_children[value], self.impurity_evaluator, self.input_data)  


    def assign_index(self, cur = 0):
        self.idx = cur
        self.max_index = cur
        for key, child in self.children.items():
           self.max_index = child.assign_index(self.max_index+1)

        return self.max_index

    def __str__(self):                  # used for printing the node, add more info if reqd
        if self.attr != '-':
            return f"Id = {self.idx}\n Attribute = {self.attr}\n Size = {len(self.indices)}\n Prediction = {self.class_value}"
        else:
            return f"Id = {self.idx}\n Size = {len(self.indices)}\n Target Value = {self.class_value}"
