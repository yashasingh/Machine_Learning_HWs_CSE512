import numpy as np
import argparse
import collections


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

    return features, labels


def initiate_centroids(features, no_of_features, k):
    centroids = dict()
    data_indx = np.random.randint(low=0, high=no_of_features, size=k)
    random_data_points = [features[i] for i in data_indx]
    for indx, data_point in enumerate(random_data_points):
        centroids[indx] = data_point
    return centroids


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


def manhattan_distance(point1, point2):
    distance = 0
    p1 = np.array(point1)
    p2 = np.array(point2)
    for i in range(len(p1)):
        distance += abs(p1[i]-p2[i])
    return distance


def kmeans_fit_data(features, centroids, k, labels, dist_method="eucledian", thresh = 0.0001, mx_iter = 10):
    """
    Main K-means clustering algirithm
    """  
    converged = False
    iteration = 0

    print("\nTotal number of clusters: ", k)

    while (not converged) and (iteration < mx_iter):
        cluster_group = dict()
        cluster_labels = dict()
        for i in range(k):
            cluster_group[i] = list()
            cluster_labels[i] = list()
        
        distances = list()
        for feature in features:
            if dist_method == "eucledian":
                distances.append([eucledian_distance(feature, centroid) for centroid in centroids.values()])
            else:
                distances.append([manhattan_distance(feature, centroid) for centroid in centroids.values()])

        feature_mapped_cluster = np.argmin(distances, axis=1)

        for indx, cluster in enumerate(feature_mapped_cluster):
            cluster_group[cluster].append(features[indx])
            cluster_labels[cluster].append(labels[indx])

        old_centroids = centroids.copy()

        for indx, cluster_data_points in enumerate(cluster_group.values()):
            centroids[indx] = np.mean(cluster_data_points, axis=0)

        converged = True
        mx_diff = thresh
        for indx, centroid in enumerate(centroids.values()):
            diff = sum(abs(old_centroids[indx]-centroid)/old_centroids[indx])
            mx_diff = max(mx_diff, diff)
            if diff > thresh:
                converged = False

        positive_diagnosis = [0 for i in range(k)]
        missclassified = 0
        for indx, feature_labels in enumerate(cluster_labels.values()):
            count = collections.Counter(feature_labels)
            if 1.0 in count.keys():
                positive_diagnosis[indx] = count[1.0]
            majority_label = count.most_common(1)[0][0]
            missclassified += len([i for i in feature_labels if i != majority_label])


        print("================= Iteration ", iteration+1,"=================")
        for indx, data_points in enumerate(cluster_labels.values()):
            print("Cluster "+str(indx+1)+" size: "+str(len(data_points)))
            positive_percent = (positive_diagnosis[indx]/len(data_points))*100
            # print("Positive diagnosis in cluster "+str(indx)+" is : "+str(positive_diagnosis[indx])+" ({0:.2f}%)\n".format(positive_percent))
            print("Positive diagnosis  : "+str(positive_diagnosis[indx])+" ({0:.2f}%)\n".format(positive_percent))

        iteration+=1
    print("")
    return centroids


def predict_cluster(features, centroids, dist_method="eucledian"):
    distances = list()
    for feature in features:
        if dist_method == "eucledian":
            distances.append([eucledian_distance(feature, centroid) for centroid in centroids.values()])

    return np.argmin(distances, axis=1)


def check_misclassified(feature_mapped_clusters, labels, k):
    groups = dict()

    for i in range(k):
        groups[i] = list()

    for indx, cluster in enumerate(feature_mapped_clusters):
        groups[cluster].append(labels[indx])
    
    missclassified = 0
    for indx, clusters in enumerate(groups.values()):
        count = collections.Counter(clusters)
        majority_label = count.most_common(1)[0][0]
        missclassified += len([i for i in clusters if i != majority_label])

    return missclassified

if __name__ == "__main__":
    """
    Main runner for this file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="Specify the path for  dataset file")
    parser.add_argument("--K", help="Specify the K parameter for number of clusters", default=2)
    parser.add_argument("--distance", help="Specify distance metric", default="eucledian")
    args = parser.parse_args()

    k = int(args.K)
    dist_method = args.distance
    dataset = args.dataset
    print(k , dist_method, dataset)
    print("Reading data")
    features, labels  = get_data(dataset)
    no_of_features = features.shape[0]

    ## Populationg initial cluster centres (Random data-points)
    centroids = initiate_centroids(features, no_of_features, k)

    ## Updating random clusters using K-means clustering algorithn
    centroids = kmeans_fit_data(features, centroids, k, labels, dist_method)
    
    ## Extracting the group cluster to which each data-point belongs
    feature_mapped_clusters = predict_cluster(features, centroids)

    ## Calculating the total missclassified samples in each cluster"
    missclassified = check_misclassified(feature_mapped_clusters, labels, k)

    print("Total missclassified samples are " + str(missclassified))

    print("END")