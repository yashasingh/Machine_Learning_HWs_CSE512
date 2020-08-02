import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets.samples_generator import make_blobs


def get_data():
    """
    Function used to create the data using sklearn make_blobs.
    The generated data is shuffled by make_blobs itself.
    """
    X0, y = make_blobs(n_samples=100, n_features = 2, centers=2, cluster_std=1.05, random_state=10)
    return (X0, y)


def get_end_fvals(X1, y):
    """
    Function to get min and max feature values in given data.
    """
    negative_x, positive_x = [], []
    for i,label in enumerate(y):
        if label == 0: negative_x.append(X1[i])
        else: positive_x.append(X1[i])
    max_feature_val = np.amax(positive_x) if np.amax(positive_x) > np.amax(negative_x) else np.amax(negative_x)
    min_feature_val = np.amin(positive_x) if np.amin(negative_x) < np.amin(negative_x) else np.amin(negative_x)
    return (max_feature_val, min_feature_val)


def change_labels(y):
    """
    Function to change all 0 labels to -1.
    Hence final classes of labels are 1 and -1.
    """
    y1 = [-1.0 if i == 0 else 1.0 for i in y]
    return np.array(y1)


def train_test_split(X, y):
    """
    Provided data is already shuffled,
    this function splits the entire data into test and train sets.
    train data and corrosponding labels constitute 80% of total data.
    test data and corrosponding labels constitute 20% of remaining data.
    """
    train_idx = int(X.shape[0] * 0.8)
    train_data = X[:train_idx]
    train_label = y[:train_idx]
    test_data = X[train_idx:]
    test_label =  y[train_idx:]
    return (train_data, train_label, test_data, test_label)


def train(iterations, train_data, train_label, max_feature_val, min_feature_val):
    """
    MAIN SVM TRAINING function using SGD.
    Weights and theta are initialized by the calculated max feature value.
    This function progressivelty decreases steps by 0.1 in every iteration.
    """
    bias  = 0
    best_weights = None
    min_misclassified = train_data.shape[0]
    for b in np.arange(min_feature_val, max_feature_val, 0.1):
      step_down_factor = 0.1
      misclassified = 0
      lmbda = max_feature_val
      weights = np.array([max_feature_val, max_feature_val]) 
      theta = np.array([max_feature_val, max_feature_val]) 
      sum_of_weights = np.zeros(train_data.shape[1])
      for i in range(1, iterations+1):
          weights = (1/i*lmbda*step_down_factor)*theta
          sum_of_weights += weights
          idx = np.random.randint(len(train_data))
          if train_label[idx] * (np.dot(train_data[idx], weights) + b) < 1:

              misclassified += 1
              theta = theta + train_label[idx]*train_data[idx]
      if misclassified < min_misclassified:
        min_misclassified = misclassified
        best_weights = sum_of_weights/iterations
        bias = b
    return  best_weights, bias


def test(test_data, test_label, weights, bias):
    """
    This function returns the predicted labels for provided test data.
    predicted values = dot procuct of weights and data-points.
    """
    y_predicted = np.array([])
    for i in range(test_data.shape[0]):
        predicted_val = np.dot(weights, test_data[i]) + bias
        predicted_sign = np.sign(predicted_val)
        y_predicted = np.append(y_predicted, predicted_sign)
    return y_predicted


def draw(data_points, labels, final_weights, bias):
    """
    Function to vistalise the sample data along with the maximum-margin separating hyperplane.
    Visualization referece provided by instructors at piazza.
    Reference: https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html
    """
    decision_func = lambda weights, bias, data: np.dot(data, weights) + bias

    plt.scatter(data_points[:, 0], data_points[:, 1], marker='o', c=labels, s=20, cmap="spring", edgecolor='k')

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xrange = np.linspace(xlim[0], xlim[1], 30)
    yrange = np.linspace(ylim[0], ylim[1], 30)
    Yrange, Xrange = np.meshgrid(yrange, xrange)
    XY = np.vstack([Xrange.ravel(), Yrange.ravel()]).T
    Zvals = decision_func(final_weights, bias, XY).reshape(Xrange.shape)
    ax.contour(Xrange, Yrange, Zvals, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])
    plt.show()


def main(no_of_iterations):
    """
    Main runner function for entire file.
    """
    print("Populating data.")
    data_points, labels = get_data()
    
    print("Data creation successfull. Extracting end feature values.")
    max_feature_val, min_feature_val = get_end_fvals(data_points, labels)

    print("Extracted max feature values is %f and extracted min feature value is %f" % (max_feature_val, min_feature_val))
    labels = change_labels(labels)

    print("Splitting the data into train and test splits where 80% data is used for training and remaining 20% for testing.")
    train_data, train_label, test_data, test_label = train_test_split(data_points, labels)

    print("Initiating SVM training with stochastic gradiend descent.")
    final_weights, bias = train(no_of_iterations, train_data, train_label, max_feature_val, min_feature_val)
    print("Training successful/.")

    print("Testing the trained model.")
    test_predictions = test(test_data, test_label, final_weights, bias)
    print("Total misclassified points = ", (test_label != test_predictions).sum())

    print("Generating the plot for visualization of sameple data with the created maximum-margin separating hyperplane.")
    draw(train_data, train_label, final_weights, bias)


if __name__ == "__main__":
    main(no_of_iterations = 800)
else:
    print("File caller is: " + __name__)
    main(no_of_iterations = 800)



