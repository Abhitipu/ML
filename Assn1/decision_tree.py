class node:
    def __init__(self, conjunction, indices, impurity_evaluator, input_data):

        '''
            attr : each attribute is identified with an index (just a string maybe construct a hash)
            conjunction: values of the attributes from the root to the parent node(currently a list! convert to a 6 digit integer)
            categories: the classValues of the inputs [nCat1, nCat2, nCat3, nCat4]
            indices: indices of the entries in the dataFrame [corresponding to the conjunction]
            impurity_evaluator is the function which will evaluate our gains for splits
            2 possibilities: Gini index / Info gain
        '''
    
        self.conjunction = conjunction                  # upto the parent
        self.indices = indices                          # a list of all the relevant indices

        self.impurity_evaluator = impurity_evaluator    # for getting the best split in construction
        
        self.children = []                              # empty list for now
        self.class_value = ''                           # useful for leaf nodes! Currently using most popular category only
        self.total = len(indices)                       # total no of nodes
        self.impurity = 0.0                             # need to compute impurity as well... is it required now?
        self.attr = ''                                  # this will be decided later
        self.categories = Dict()                        # contains the classifications of input
        self.input_data = input_data                    # This should rather be static!

        self.compute_values()                           # compute initial values
        self.propagate()                                # extend the tree
    
    def compute_values():
        '''
            We are finding out the classifications of our input data
        '''
        self.categories = my_data.get_classification(indices)

        self.best_frequency = 0
        for key, value in self.categories.items():
            if value >= self.best_frequency:
                self.est_frequency = value
                self.class_value = key


    def propagate(self):

        '''
            Now we have to construct it further
            We can stop if we have a pure split i.e. all out data points are in 1 category
            Or we have exhausted all attributes
        '''
        
        if self.best_frequency == self.total:       # pure node
            return
        
        done = True
        for attribute in attr_list:                 # if we haven't exhausted all attributes
            if self.conjunction[attribute] == '':
                done = False

        if done:
            return
        
        max_gain = 0.0
        best_attribute = ''
        base_impurity = self.impurity_evaluator(self.categories)        # advantage: both gini gain and info gain are calculated in the same manner

        for attribute in attr_list:
            if self.conjunction[attribute] != '':                       # attribute is already chosen
                continue
            else:
                childrenFreq = my_data.get_split(attribute, indices)    # a dictionary of the nodes in the corresponding subtree
                
                curr_gain = base_impurity                               

                for key in childrenFreq:
                    new_classification = my_data.get_classification(children_freq[key])
                    curr_gain -= (len(children_freq[key])/self.total)*self.impurity_evaluator(new_classification)   # subtract the weighted mean
                    
                if curr_gain >= max_gain:                               # re assign if necessary
                    max_gain = curr_gain
                    best_attribute = attribute

        new_conjunction = self.conjunction
        my_children = my_data.get_split(best_attribute, indices)
        self.attr = best_attribute                                      # Set the attribute for the node

        for value in possible_values[best_attribute]:   
            new_conjunction[best_attribute] = value
            new_node = node(new_conjunction, my_children[value], self.impurity_evaluator, self.input_data)  # add to child list
            self.children.append(new_node)
