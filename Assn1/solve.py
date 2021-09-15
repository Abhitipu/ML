import pandas as pd
import math
from graphviz import Digraph

from decision_tree import node
from description import attr_list, possible_values, classifications, header_list
from data_handling import my_data


def gini_index(categories):
    tot = 0
    for key in categories:
        tot += categories[key]
    
    if tot == 0:
        return 0

    value = 1.0
    for key in categories:
        frac = categories[key] / tot
        value -= frac*frac

    return value

def entropy(categories):
    tot = 0
    for key in categories:
        tot += categories[key]

    if tot == 0:
        return 0

    value = 0.0
    for key in categories:
        if(categories[key] == 0):
            continue
        frac = categories[key] / tot
        value -= frac*math.log2(frac)

    return value

def construct_tree(my_input):
    my_conjunction = dict()
    for attribute in attr_list:
        my_conjunction[attribute] = ''
    
    my_input.gen_test_and_validation_set()                              # generate random test and validation set!
    my_indices = [i for i in range(len(my_input.training_set))]         # constructing on the training set
    
    root1 = node(my_conjunction, my_indices, entropy, my_input)         # 2 trees are constructed
    root2 = node(my_conjunction, my_indices, gini_index, my_input)      # using the 2 impurity measure functions

    return root1, root2

def compute_accuracy():
    pass

def get_depth_limit():
    pass

def prune_tree():
    pass

def print_tree(my_tree):
    
    my_graph= Digraph('Decision Tree', filename='./output_files/decision_tree.gv')
    my_graph.attr(rankdir='LR', size='1000,500')

    my_graph.attr('node', shape='rectangle')
    
    # doing a bfs using a queue
    qq = [my_tree]  # using a list as a queue for the bradth first search
    while len(qq) > 0:
        node = qq.pop(0)         
        for key, child in node.children.items():
            my_graph.edge(str(node), str(child), label=key)
            qq.append(child)

    my_graph.render('./output_files/decision_tree.gv', view=True)

if __name__ == "__main__":
    my_input = my_data('input_files/car.data')
    
    tree1, tree2 = construct_tree(my_input)

    compute_accuracy()
    get_depth_limit()
    prune_tree()

    print_tree(tree1)
    # print_tree(tree2)
