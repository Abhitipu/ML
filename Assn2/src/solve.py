from data_handling import my_data
from sklearn.neighbors import KNeighborsClassifier

if __name__ == "__main__":
    my_dataset = my_data('../input_files/spam_ham_dataset.csv') # reading from csv


    for num_nbrs in range(1, 25):
        my_dataset.gen_test_and_validation_set()

        neigh = KNeighborsClassifier(n_neighbors = num_nbrs)
        neigh.fit(my_dataset.training_set, my_dataset.training_set_labels)

        model_predictions = neigh.predict(my_dataset.validation_set)

        # print(type(model_predictions))
        # print(type(my_dataset.validation_set_labels))
        cnt = 0

        for i in range(len(model_predictions)):
            if model_predictions[i] == my_dataset.validation_set_labels[i]:
                cnt += 1

        print(f"Accuracy for {num_nbrs} neighbors is {cnt/len(model_predictions)}")

