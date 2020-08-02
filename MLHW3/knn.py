import numpy as np
import collections
import argparse


def get_data(file_name):
    """
    Function to load the given data.
    """
    data = np.genfromtxt(file_name, delimiter=',')
    ## Seperating out features ##
    columns = len(data[0, :])
    features = data[1:,0:columns-1]
    ## Seperating out Labels ##
    labels = data[1:, -1]
    return (features, labels)


def split_data(X, y):
    """
    Provided data is already shuffled, splitting dataset into 80:20 train-test ratio.
    """
    train_idx = int(X.shape[0] * 0.8)
    train_data = X[:train_idx]
    train_label = y[:train_idx]
    test_data = X[train_idx:]
    test_label =  y[train_idx:]
    return (train_data, train_label, test_data, test_label)


def eucledian_distance(point1, point2):
    """
    Calculates and returns the Eucledian distance between two vectors.
    """
    distance = 0
    p1 = np.array(point1)
    p2 = np.array(point2)
    for i in range(len(p1)):
        distance += (p1[i]-p2[i])**2
    distance = np.sqrt(distance)
    return distance


def train(X,y, X_test):
    """
    Finds and returns the distance of test vector with all points in the train dataset.
    """
    all_distance = list()
    for i,item in enumerate(X):     # Append the tuple - distance between train and test point along with label of train point
        all_distance.append((eucledian_distance(item, X_test), y[i]))
    all_distance.sort(key = lambda x: x[0])
    return all_distance


def get_majority_label(K, sorted_distances):
    """
    Finds the  class of majoirity of K-neighbors.
    """
    mindist_label_set  = list()
    K = int(K)
    for i in range(K):
        mindist_label_set.append(sorted_distances[i][1])
    count = collections.Counter(mindist_label_set)
    return count.most_common(1)[0][0]


if __name__ == "__main__":
    """
    Main runner for this file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Specify the path for  dataset file")
    parser.add_argument("--K", help="Specify the K parameter")
    args = parser.parse_args()

    print("Reading data")
    features, labels = get_data(args.dataset)

    print("Splitting the data into train and test splits where 80% data is used for training and remaining 20% for testing")
    train_X, train_y, test_X, test_y = split_data(features, labels)

    predictions = list()
    misclassifications = 0
    classification_score = 0

    ## Find majority label for each test point ##
    for i, test_item in enumerate(test_X):
        sorted_dist = train(train_X, train_y, test_item)
        predictions.append(get_majority_label(args.K, sorted_dist))
        if predictions[i] != test_y[i]:
            misclassifications += 1
        classification_score = 1 - (misclassifications/len(test_y))

    print("classification_score :", classification_score)
