# This code is an adaptation of the code published in:
# https://github.com/lazyprogrammer/machine_learning_examples/

from process import get_binary_data
import numpy as np

# Getting the binary data from the dataset
X, Y = get_binary_data()

# Saving the dimensionality of the dataset
DIMENSIONS = X.shape[1]

# Initialize the weights of our logistic regression model
W = np.random.randn(DIMENSIONS)

# No bias
b = 0

# Sigmoid function
def sigmoid(z):
	return 1/(1+np.exp(-z))

# This return the Sigmoid of the dot product between the variables and the weights plus bias, which will be the untrained probability of Y given X.
def forward(X, W, b):
	return sigmoid(X.dot(W)+b)

# Untrained probabilities
P_Y_given_X = forward(X, W, b)

# Rounding the probabilities
predictions = np.round(P_Y_given_X)

# This function takes in input the gold and the predictions, returning the accuracy (number in which gold and predictions are equal divided by the total)
def classification_rate(Y, P):
	return np.mean(Y == P)

# Print the score
print("Score: ", classification_rate(Y, predictions))