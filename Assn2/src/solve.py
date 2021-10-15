import matplotlib.pyplot as plt
from data_handling import my_data
from knn_model import knn_classifier


def run_model():
    pass

def vary_num_nbrs():
    pass

def plot(x_values, y_values, filename):
    plt.plot(x_values, y_values)
    plt.savefig(filename)
    return

if __name__ == "__main__":

    # reading from csv
    my_dataset = my_data('../input_files/spam_ham_dataset.csv') 
    
    #generating a random test and validation set
    my_dataset.gen_test_and_validation_set()
    
    run_model()
    vary_num_nbrs()
    plot()
