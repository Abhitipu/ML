import pandas as pd
import math
from decision_tree import node
from description import attr_list, possible_values, classifications

def construct_tree(my_input):
    my_conjunction = Dict()
    for attribute in attr_list:
        my_conjunction[attribute] = ''

    my_indices = [i for i in range(my_data.df.size)]        # maybe choose some random indices here... 80/20 splits

    root1 = node(my_conjunction, my_indices, gini_gain)     # 2 trees are constructed
    root2 = node(my_conjunction, my_indices, gain_ratio)    # using the 2 impurity measure functions

    return root1, root2

def compute_accuracy():
    pass

def get_depth_limit():
    pass

def prune_tree():
    pass

def print_tree():
    pass

def gini_index(categories):
    tot = 0
    for key in categories:
        tot += categories[key]
    
    value = 1.0
    for key in categories:
        frac = categories[key] / tot
        value -= frac*frac

    return value

def entropy(categories):
    tot = 0
    for key in categories:
        tot += categories[key]
    
    value = 0.0
    for key in categories:
        frac = categories[key] / tot
        value -= frac*math.log2(frac)

    return value

if __name__ == "__main__":
    my_input = my_data('input_files/car.data')

    tree1, tree2 = construct_tree(my_input)

    compute_accuracy()
    get_depth_limit()
    prune_tree()
    print_tree()
