import numpy as np
import argparse

import matplotlib.pyplot as plt

max_iter = 3
# max_iter = 1000

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


def adaboost(fea, labels, iterations=max_iter):

	config = dict(list())   # Dictionary to store best classifier congfiguration in each iteration

	rows, cols = np.shape(fea)

	weight_vector = np.repeat((1/rows), rows)

	############  TRAINING   ################
	
	for T in range(1, iterations + 1):

		least_error, best_threshold, p, best_feature = float('inf'), float('inf'), 1, -1
		confidence, best_confidence = 0.0, 0.0
		best_predictions = np.ones(len(labels))

		# Iterating over individual feature to determine best classifier for our data

		for f in range(cols):

			curr_feature = fea[:, f]

			all_thresholds = np.unique(curr_feature)

			for threshold in all_thresholds:
				error = 0.0
				polarity = 1

				prediction_col = np.ones(len(labels))
				prediction_col[np.where(fea[:, f] < threshold)] = -1

				error = np.sum(weight_vector[labels != prediction_col])

				if error > 0.5:
					error = 1 - error 
					polarity = -1
					prediction_col = np.ones(len(labels))
					prediction_col[np.where(fea[:, f] >= threshold)] = -1


				if error < least_error:
					least_error = error
					best_threshold = threshold
					best_feature = f
					best_polarity = polarity
					best_predictions = prediction_col

				# print('#iter', error)

		# print("Error :  inside", least_error, best_threshold)

		# Find the confidence 

		confidence = (1/2)*np.log((1.0/least_error)-1)
		if confidence > best_confidence:
			best_confidence = confidence

		config[T] = [best_confidence, best_polarity, best_feature, best_threshold]

		# Update the weights

		for i in range(len(weight_vector)):
			weight_vector[i] = weight_vector[i]*(np.exp(-1 * best_confidence * best_predictions[i] * labels[i]))

		# Normalize the weight vector

		weight_vector = weight_vector/np.sum(weight_vector) 

	print("Weights : ", weight_vector)
	return config


def testing(all_configs, X_test, Y_test):
	final_predicted_label = []

	for data in X_test:

		weighted_confidence = 0

		for key in all_configs:
			confidence, p, feature_idx, thres = all_configs[key]

			val = data[feature_idx]
			if val < thres:
				val = -1 * p
			else:
				val = p

			weighted_confidence += confidence * val

		final_predicted_label.append(np.sign(weighted_confidence))

	error = sum([
			1.0 
			for i in range(len(Y_test))
			if final_predicted_label[i] != Y_test[i] 
		]
	)/len(Y_test)

	return error
	


def k_fold(X, labels, K):
	for k in range(K):
		training = np.array([x for i, x in enumerate(X) if i % K != k])
		training_labels = np.array([x for i, x in enumerate(labels) if i % K != k])
		validation = np.array([x for i, x in enumerate(X) if i % K == k])
		validation_labels = np.array([x for i, x in enumerate(labels) if i % K == k])
		yield training, training_labels, validation, validation_labels, k


def cross_validate(features, labels, to_print=True, iter_cnt=max_iter, num_folds=10):
	all_error_fold = 0.0

	for training, training_labels, validation, validation_labels, k in k_fold(features, labels, num_folds):
		configs = adaboost(training, training_labels, iter_cnt)
		fold_error = testing(configs, validation, validation_labels)

		if to_print:
			print("Error after fold %d : %f" % (k+1, fold_error))
		all_error_fold += fold_error

	val_error = all_error_fold/num_folds
	if to_print:
		print("MEAN ERROR AVERAGED OVER 10 FOLDS  :  ", val_error)

	return val_error


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument("--dataset", help="Specify the path for  dataset file")

	parser.add_argument("--mode", help= "Choose mode - erm or cv or plots")	

	args = parser.parse_args()

	features, labels = get_data(args.dataset)


	if args.mode == "erm":
		configs = adaboost(features, labels)
		error = testing(configs, features, labels)
		print(" ERROR : " , error)


	if args.mode == "cv":
		cross_validate(features, labels)

	if args.mode == "plots":
		validation_errors = []
		training_errors = []

		# iter_ranges = range(1, 501)
		iter_ranges = range(1, 18)

		for iter_cnt in iter_ranges:
			model = adaboost(features, labels, iter_cnt)
			training_errors.append(testing(model, features, labels))
			validation_errors.append(cross_validate(features, labels, False, iter_cnt))

		plt.plot(iter_ranges, training_errors, label="Training Errors", marker='x')
		plt.plot(iter_ranges, validation_errors, label="Validation Errors", marker='o')

		plt.title("Training vs. Validation Errors (Different T values)")

		plt.xlabel("Max. Iterations")
		plt.ylabel("Error")

		plt.legend()
		plt.show()