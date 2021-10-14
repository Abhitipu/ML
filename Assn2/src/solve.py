import matplotlib.pyplot as plt
from data_handling import my_data
from sklearn.neighbors import KNeighborsClassifier

if __name__ == "__main__":
    my_dataset = my_data('../input_files/spam_ham_dataset.csv') # reading from csv

    my_dataset.gen_test_and_validation_set()
    k_nbrs = []
    accuracy_values = []
    
    for num_nbrs in range(1, 4102, 50):

        neigh = KNeighborsClassifier(n_neighbors = num_nbrs)
        neigh.fit(my_dataset.training_set, my_dataset.training_set_labels)

        model_predictions = neigh.predict(my_dataset.validation_set)

        cnt = 0

        for i in range(len(model_predictions)):
            if model_predictions[i] == my_dataset.validation_set_labels[i]:
                cnt += 1

        k_nbrs.append(num_nbrs)
        accuracy_values.append(cnt/len(model_predictions))

    plt.plot(k_nbrs, accuracy_values)
    plt.savefig('../output_files/num_nbrs_vs_accuracy')
