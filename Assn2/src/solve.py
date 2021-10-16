import time
import matplotlib.pyplot as plt
from data_handling import my_data
from knn_model import knn_classifier
from distance_functions import cosine_similarity, manhattan_distance, euclidian_distance

def run_model(my_dataset):
    my_classifier = knn_classifier(my_dataset.training_set, my_dataset.validation_set, my_dataset.training_set_labels, my_dataset.validation_set_labels, euclidian_distance)
    my_classifier.predict()
    accuracy = my_classifier.compute_accuracy()
    print(f"Accuracy obtained is {accuracy}")
    return


def vary_num_neighbours(my_dataset, lower_bound, upper_bound, delta):
    my_functions = [cosine_similarity, manhattan_distance, euclidian_distance]

    for my_func in my_functions:
        print(f"Using {my_func.__name__} for distances")
        number_of_nbrs = []
        accuracies = []

        for k in range(lower_bound, upper_bound, delta):
            my_classifier = knn_classifier(my_dataset.training_set, my_dataset.validation_set, my_dataset.training_set_labels, my_dataset.validation_set_labels, euclidian_distance, k)
            my_classifier.predict()
            accuracy = my_classifier.compute_accuracy()

            number_of_nbrs.append(i)
            accuracies.append(accuracy) 

            print(f"Accuracy obtained for {i} neighbors is {accuracy}") 

        plot(number_of_nbrs, accuracies, my_func.__name__)

    return

def plot(x_values, y_values, filename):
    '''
        This will be called from the run model and the vary_num_neighbours functions
        This will plot and save in the output_files folder
    '''
    plt.plot(x_values, y_values)
    plt.xlabel("Number of neighbors")
    plt.ylabel("Accuracy values")
    plt.savefig(f"../output_files/{filename}_num_nbrs_vs_accuracy")
    return

def showtime(start_time):
    print(f"Time taken: {time.time() - start_time}")
    return

if __name__ == "__main__":

    start = time.time()
    # reading from csv
    my_dataset = my_data('../input_files/spam_ham_dataset.csv') 
    print("Read data")
    showtime(start)
    
    #generating a random test and validation set
    my_dataset.gen_test_and_validation_set()
    
    run_model(my_dataset)
    # vary_num_neighbours(my_dataset, 100, 4100, 50)
    print("Ran model")
    showtime(start)
