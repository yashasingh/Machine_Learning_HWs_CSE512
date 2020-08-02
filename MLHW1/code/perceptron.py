import numpy as np
import argparse
import os
import matplotlib.pyplot as plt


def get_data(file_name):
    
    # data = genfromtxt('linearly-separable-dataset.csv', delimiter=',')

    data = np.genfromtxt(file_name, delimiter=',')

    # Seperating out features 

    columns = len(data[0, :])
    features = data[1:,0:columns-1]

    # Seperating out Labels

    labels = data[1:, -1]
    labels = [ 1 if i == 1 else -1 for i in labels ]

    return (features, labels)


def perceptron(training, training_labels, mode):
    all_train_error_list = []
    initial_w = np.asarray([0.0 for i in range(training.shape[1] + 1)]) # defined initial weight

    bias = 1
    keep_running = True
    loop_count = 0
    max_iter = 1000

    train_error, prev_error = 0, 1

    # TRAINING
    while loop_count < max_iter:
        loop_count += 1

        for ind, row in enumerate(training):
            predicted_value = np.dot(initial_w[1:], row) + initial_w[0]

            if predicted_value <= 0:
                predicted_label = -1
            else:
                predicted_label = 1

            if predicted_label * training_labels[ind] <= 0:
                new_w = initial_w
                new_w[1:] = initial_w[1:] + (training_labels[ind]) * row
                new_w[0] += training_labels[ind]
                initial_w = new_w

        
        train_error = 0
        for ind, row in enumerate(training):
            predicted_value = np.dot(initial_w[1:], row) + initial_w[0]

            if predicted_value <= 0:
                predicted_label = -1
            else:
                predicted_label = 1

            if predicted_label * training_labels[ind] <= 0:
                train_error += 1

        train_error = train_error / len(training)

        if mode != "cv":
            # print("Train Error after epoch {0}: {1}".format(loop_count, train_error))
            all_train_error_list.append(train_error)

        # Compare with previous error for convergence
        # if np.abs(train_error - prev_error) <= threshold:
        #   keep_running = False

        if train_error == 0.0:
            break

        prev_error = train_error

    if mode != "cv":
        print("Train Error: {0}".format(train_error))
        print("Weights: ", initial_w)
        return all_train_error_list
 
    return initial_w


def k_fold(X, labels, K):
    for k in range(K):
        training = np.array([x for i, x in enumerate(X) if i % K != k])
        training_labels = np.array([x for i, x in enumerate(labels) if i % K != k])
        validation = np.array([x for i, x in enumerate(X) if i % K == k])
        validation_labels = np.array([x for i, x in enumerate(labels) if i % K == k])
        yield training, training_labels, validation, validation_labels, k


def perceptron_with_kfold(features, labels):

    num_folds = 10
    threshold = 0.01

    all_error_fold = []

    # ITERATION FOR K-FOLD
    for training, training_labels, validation, validation_labels, k in k_fold(features, labels, num_folds):
        # TRAINING
        initial_w = perceptron(training, training_labels, "cv")


        # VALIDATION
        error = 0
        for j, validation_point in enumerate(validation):
            prediction = np.dot(initial_w[1:], validation_point) + initial_w[0]

            if prediction <=0:
                prediction = -1
            else:
                prediction = 1

            if validation_labels[j] != prediction:
                error += 1
          
    
        fold_err = error/len(validation)

        print("Error after fold %d  :    %f" % (k+1, fold_err))
        print("Weights : ", initial_w)

        all_error_fold.append(fold_err)

    print("MEAN ERROR AVERAGED OVER 10 FOLDS  :  ", sum(all_error_fold)/len(all_error_fold))

    return all_error_fold

# def plot_graph(all_train_error_list):
#     x = list(range(len(all_train_error_list)))
#     y = all_train_error_list
#     plt.plot(x, y)
#     plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", help="Specify the path for  dataset file")
    parser.add_argument("--mode", help= "Choose mode - erm or cv")

    args = parser.parse_args()

    features, labels = get_data(args.dataset)

    if args.mode == "cv":
        all_fold_error = perceptron_with_kfold(features, labels)
        # plot_graph(all_fold_error)
    if args.mode == "erm":
        all_train_error_list = perceptron(features, labels, args.mode)
        # plot_graph(all_train_error_list)
