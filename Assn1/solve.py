import pandas as pd
from graphviz import Digraph
import time
import math

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

    if tot == 0:                # check this... should never hit... maybe an assertion would be good here
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

def calc_score(Y, Y_pred):
    cnt = 0
    for i, val in enumerate(Y):
        if val == Y_pred[i]:
            cnt += 1
    accuracy = cnt/len(Y)
    return accuracy*100

def compute_accuracy(my_input, my_tree):
    avg_acc = 0
    for i in range(10):
        my_input.gen_test_and_validation_set()
        X_data = my_input.validation_set
        preds = my_tree.predict_value(X_data)
        data = []
        for index, row in X_data.iterrows():
            data.append(row['Class_value'])
        acc = calc_score(data, preds)
        avg_acc += acc

    return avg_acc/10

def get_depth_limit(my_tree):
    X_data = my_input.validation_set
    data = []
    accuracy_values = []
    for index, row in X_data.iterrows():
        data.append(row['Class_value'])

    best_accuracy = 0
    best_height = -1
    for height in range(1, len(attr_list) + 2):
        preds = my_tree.predict_value(X_data, height)
        acc = calc_score(data, preds)
        accuracy_values.append(acc)
        if acc > best_accuracy:
            best_accuracy = acc
            best_height = height
    
    '''
        TODO: plot graph
    '''
    return best_height, best_accuracy

def prune_tree(my_tree):
    '''
        Here we'll have to execute the while loop
        and make a call to prune the tree
    '''
    
    while(True):
        check_vals = dict()
        my_tree.prune_tree(check_vals)
        
        best_improvement = 0
        best_node = -1

        for key, value in check_vals.items():
            if value > best_improvement:
                best_improvement = value
                best_node = key

        if best_node == -1:
            break
        
        my_tree.remove_node(best_node)

    return 

def print_tree(my_tree, op_file):
    
    my_graph= Digraph('Decision Tree', filename=op_file)
    my_graph.attr(rankdir='LR', size='1000,500')

    my_graph.attr('node', shape='rectangle')
    
    # doing a bfs using a queue
    qq = [my_tree]  # using a list as a queue for the bradth first search
    while len(qq) > 0:
        node = qq.pop(0)         
        for key, child in node.children.items():
            my_graph.edge(str(node), str(child), label=key)
            qq.append(child)

    my_graph.render(op_file, view=True)

    return

if __name__ == "__main__":

    print("Reading from csv....")
    start = time.time()

    my_input = my_data('input_files/car.data')
    # Part 1

    print("Constructing both decision trees")
    tree1, tree2 = construct_tree(my_input)
    print("Done")
    print(f"Time taken: {time.time()-start} seconds\n")
    
    print("Computing accuracy of tree1")
    accuracy1 = compute_accuracy(my_input, tree1)
    print(f"Done, Got accuracy {accuracy1}")

    print("Computing accuracy of tree2")
    accuracy2 = compute_accuracy(my_input, tree2)
    print(f"Done, Got accuracy {accuracy2}")

    print(f"Time taken: {time.time()-start} seconds\n")

    better_tree = tree2
    if accuracy1 > accuracy2:
        better_tree = tree1
    
    print("Evaluating best depth limit")
    best_height, best_accuracy = get_depth_limit(better_tree)
    print(f"Done, Best depth limit = {best_height}, Best accuracy = {best_accuracy}")

    print(f"Time taken: {time.time()-start} seconds\n")
    
    '''
    # Part 4
    prune_tree(tree1)
    prune_tree(tree2)

    # Part 5
    print_tree(tree1, './output_files/decision_tree_entropy.gv')
    print_tree(tree2, './output_files/decision_tree_gini_index.gv')
    '''
