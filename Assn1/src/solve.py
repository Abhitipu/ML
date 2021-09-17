from graphviz import Digraph
import copy
import time
import matplotlib.pyplot as plt

from decision_tree import node
from description import attr_list, possible_values, classifications, header_list
from data_handling import my_data
from impurity_calculators import gini_index, entropy

def construct_tree(my_dataset, impurity_calculator):
    my_conjunction = dict()
    for attribute in attr_list:
        my_conjunction[attribute] = ''
    
    my_dataset.gen_test_and_validation_set()                              # generate random test and validation set!
    my_indices = [i for i in range(len(my_dataset.training_set))]         # constructing on the training set
    
    tree = node(my_conjunction, my_indices, impurity_calculator, my_dataset)         # 2 trees are constructed
    return tree

def calc_score(Y, Y_pred):
    cnt = 0
    for i, val in enumerate(Y):
        if val == Y_pred[i]:
            cnt += 1
    accuracy = cnt/len(Y)
    return accuracy*100

def compute_accuracy(my_dataset, impurity_calculator):
    '''
        Return accuracy, best tree and the corresponding validation set
    '''

    best_tree = None
    best_validation_set = None
    avg_acc = 0
    best_accuracy = 0

    for i in range(10):

        cur_tree = construct_tree(my_dataset, impurity_calculator)

        X_data = my_dataset.validation_set
        preds = cur_tree.predict_value(X_data)

        data = []
        for index, row in X_data.iterrows():
            data.append(row['Class_value'])
        acc = calc_score(data, preds)

        if acc > best_accuracy:
            best_accuracy = acc
            best_tree = cur_tree
            best_validation_set = X_data

        avg_acc += acc

    avg_acc /= 10

    return avg_acc, best_tree, best_validation_set, best_accuracy

def get_depth_limit(my_tree, my_validation_set):
    X_data = my_validation_set
    data = []
    accuracy_values = []
    height_values = []
    for index, row in X_data.iterrows():
        data.append(row['Class_value'])

    best_accuracy = 0
    best_height = -1
    for height in range(1, len(attr_list) + 2):
        preds = my_tree.predict_value(X_data, height)
        acc = calc_score(data, preds)
        accuracy_values.append(acc)
        height_values.append(height)
        if acc > best_accuracy:
            best_accuracy = acc
            best_height = height
    
    plt.plot(height_values, accuracy_values)
    plt.xlabel('Height of the tree')
    plt.ylabel('Accuracy in validation set')
    plt.savefig('../output_files/height_vs_accuracy.png')

    return best_height, best_accuracy

def prune_tree(my_tree, curr_accuracy, validation_set):

    '''
        Here we will be using the reduced error pruning method.
        It is basically a post pruning method.
        
        pseudo code:
        while(accuracy_for_validation_set_doesnt_decrease):
            check_accuracy_after removing each non_leaf_node
            remove the one that improves accuracy the most
    
    '''

    locations = dict()
    my_tree.get_locations(locations)

    X_data = validation_set
    data = []
    for index, row in X_data.iterrows():
        data.append(row['Class_value'])
    
    new_accuracy = curr_accuracy
    
    num_nodes = []
    accuracy_values = []

    num_nodes.append(my_tree.max_index)
    accuracy_values.append(curr_accuracy)

    while(True):
        best_node = -1
        for idx in range(my_tree.max_index + 1):
            if(locations[idx].is_leaf):
                continue
            else:
                locations[idx].alter_prune()
                preds = my_tree.predict_value(X_data)
                curr_acc = calc_score(data, preds)
                if curr_acc >= new_accuracy:
                    new_accuracy = curr_acc
                    best_node = idx
                
                locations[idx].alter_prune()

        if best_node == -1:
            break
        
        cur_nodes = num_nodes[-1] - 1
        num_nodes.append(cur_nodes)
        accuracy_values.append(new_accuracy)

        print(f"Removing node: {best_node}")
        locations[best_node].alter_prune()

    plt.plot(num_nodes, accuracy_values)
    plt.xlabel('Number of nodes')
    plt.ylabel('Accuracy in validation set')
    plt.savefig('../output_files/num_nodes_vs_accuracy.png')

    return new_accuracy

def print_tree(my_tree, op_file):
    
    my_graph= Digraph('Decision Tree', filename=op_file)
    my_graph.attr(rankdir='LR', size='1000,500')

    my_graph.attr('node', shape='rectangle')
    
    # doing a bfs using a queue
    qq = [my_tree]                          # using a list as a queue for the bradth first search
    while len(qq) > 0:
        node = qq.pop(0)         
        for key, child in node.children.items():
            my_graph.edge(str(node), str(child), label=key)
            qq.append(child)

    my_graph.render(op_file, view=True)

    return

if __name__ == "__main__":

    print("Reading from csv....\n")
    start = time.time()

    my_dataset = my_data('../input_files/car.data')

    # Part 1 and 2
    print("Constructing decision tree using entropy and information gain")
    accuracy1, tree1, my_validation_set1, best_accuracy1 = compute_accuracy(my_dataset, entropy)
    print(f"Done, Got avg accuracy {accuracy1} and best accuracy {best_accuracy1}\n")

    print("Constructing decision tree using gini index and gini gain")
    accuracy2, tree2, my_validation_set2, best_accuracy2 = compute_accuracy(my_dataset, gini_index)
    print(f"Done, Got avg accuracy {accuracy2} and best accuracy {best_accuracy2}\n")

    print(f"Time taken: {time.time()-start} seconds\n")

    better_tree = tree2
    validation_set = my_validation_set2

    if accuracy1 > accuracy2:
        better_tree = tree1
        validation_set = my_validation_set1
    
    # Part 3
    print("Evaluating best depth limit")
    best_height, best_accuracy = get_depth_limit(better_tree, validation_set)
    print(f"Done, Best depth limit = {best_height}, Best accuracy = {best_accuracy}")
    print(f"Time taken: {time.time()-start} seconds\n")
    
    # Part 4
    print("Pruning tree")
    new_accuracy = prune_tree(better_tree, best_accuracy, validation_set)         # Confirm this!
    print(f"Done! Accuracy = {new_accuracy}")
    print(f"Time taken: {time.time()-start} seconds\n")

    # Part 5
    print_tree(better_tree, '../output_files/decision_tree.gv')
