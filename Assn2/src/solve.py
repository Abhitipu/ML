import time
import sys
import matplotlib.pyplot as plt
from data_handling import my_data
from knn_model import knn_classifier
from distance_functions import cosine_similarity, manhattan_distance, euclidian_distance

def run_model(my_dataset):
    '''
        This will run the model with the different difference/similarity measures
        Then it will also plot the different accuracies
    '''
    functions = [cosine_similarity, manhattan_distance, euclidian_distance]
    
    for my_func in functions:
        print(f"\nUsing {my_func.__name__}")
        my_classifier = knn_classifier(my_dataset.training_set, my_dataset.validation_set, my_dataset.training_set_labels, my_dataset.validation_set_labels, my_func)
        my_classifier.predict()
        accuracies, num_nbrs = my_classifier.compute_accuracy()
        plot(num_nbrs, accuracies, my_func.__name__)

    return


def plot(x_values, y_values, filename):
    '''
        This will be called from the run model and the vary_num_neighbours functions
        This will plot and save in the output_files folder
    '''
    plt.plot(x_values, y_values)
    plt.xlabel("Number of neighbors")
    plt.ylabel("Accuracy values")
    plt.title(filename)
    plt.savefig(f"../output_files/{filename}_num_nbrs_vs_accuracy_weighted")
    plt.cla()       # clear axes
    return

def showtime(start_time):
    print(f"Time taken: {time.time() - start_time} seconds", file=sys.stderr)
    return

if __name__ == "__main__":

    start = time.time()
    # reading from csv
    my_dataset = my_data('../input_files/spam_ham_dataset.csv') 
    print("Read data")
    showtime(start)
    
    #generating a random test and validation set
    my_dataset.gen_test_and_validation_set()
    
    # Running our code
    run_model(my_dataset)
    print("Done")
    showtime(start)
