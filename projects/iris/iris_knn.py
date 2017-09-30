# -*- coding: utf-8 -*-
"""KNN clustering to classify iris flowers

This code was written by Enrico Santus, following the examples in:

	"Introduction to Machine Learning with Python"
	by: Andrea C. Muller and Sarah Guido

It is meant to be used for training purposes only.

"""

# Importing numpy and sklearn modules for array and machine learning goals
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import matplotlib.pyplot as plt

def main():

	# Loading the dataset
	iris = load_iris()

	# Printing some info about the dataset
	print_info(iris)

	# Splitting the dataset in training and test set
	X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'], random_state=0)
	print("\n\nThe dataset has been split.\n")
	print("X_train length: " + str(X_train.shape))
	print("X_test length: " + str(X_test.shape))

	# Plotting all combinations of the dataset variables to identify patterns
	plot_data(iris, X_train, y_train)

	# Using KNN with the number of neighbors equal to 1
	knn = KNeighborsClassifier(n_neighbors=1)

	# Fitting the training data
	knn.fit(X_train, y_train)

	# Creating a test array, just to try the model and predicting its category
	X_new = np.array([[5, 2.9, 1, 0.2]])
	prediction = knn.predict(X_new)
	print("Prediction type and shape: " + str(type(prediction)) + " " + str(prediction.shape))
	print("Prediction: " + str(iris["target_names"][prediction]))

	# Testing the testset and printing the results
	y_pred = knn.predict(X_test)
	print("Test set prediction accuracy: " + str(np.mean(y_pred == y_test)))
	print("KNN score method: " + str(knn.score(X_test, y_test)))



def plot_data(iris, X_train, y_train):
	""" This function plots iris dataset, combining every pair
	of data, to allow the visual identificaiton of patterns

		Args:
			iris: iris dataset
			X_train: training split
			y_train: training gold

		Returns:
			Nothing. It prints the plot.
	"""

	fig, ax = plt.subplots(3, 3, figsize=(5, 5))
	plt.suptitle("iris_pairplot")

	for i in range(3):
		for j in range(3):
			ax[i, j].scatter(X_train[:, j], X_train[:, i + 1], c=y_train, s=60)
			ax[i, j].set_xticks(())
			ax[i, j].set_yticks(())
			if i == 2:
				ax[i, j].set_xlabel(iris['feature_names'][j])
			if j == 0:
				ax[i, j].set_ylabel(iris['feature_names'][i + 1])
			if j > i:
				ax[i, j].set_visible(False)
	plt.show()


def print_info(iris):
	""" This function prints information about the iris dataset.

		Args:
			iris: iris dataset
		Returns:
			nothing.
	"""


	print("Type: " + str(type(iris)))

	print("Keys: " + ", ".join(iris.keys()))

	print("Target Names: " + ", ".join(list(iris["target_names"])) + "\n\tType: " + str(type(iris["target_names"])) + "\n\tShape: " + str(iris["target_names"].shape))

	print("Feature Names: " + ", ".join(list(iris["feature_names"])) + "\n\tType: " + str(type(iris["feature_names"])) + "\n\tShape: " + str(len(iris["feature_names"])))

	print("Description Example:\n\n" + iris['DESCR'][:250] + "\n\n")

	print("Data: " + str(iris["data"][:5]) + "\n\tType: " + str(type(iris["data"])) + "\n\tShape: " + str(iris["data"].shape))

	print("Gold Target: " + str(iris["target"][:50]) + "\n\tType: " + str(type(iris["target"])) + "\n\tShape: " + str(iris["target"].shape))


main()
