# This code is an adaptation of the code published in:
# https://github.com/lazyprogrammer/machine_learning_examples

import numpy as np

ITEMS = 10
DIMENSIONS = 2

# Create NxD normally distributed data matrix
data = np.random.randn(ITEMS, DIMENSIONS)

# Adding a bias vector of ones to shift the activation curve to the right
# See: https://stackoverflow.com/questions/2480650/role-of-bias-in-neural-networks/2499936#2499936
bias = np.ones((ITEMS, 1)) #EQUAL TO: ones = np.array([[1]*N]).T

# Concatenating the vector to the data matrix
data_bias = np.concatenate((bias, data), axis=1)

# Creating a weight vector
weight = np.random.randn(DIMENSIONS+1) # DIMENSIONS + bias

# Calculating the Dot Product (i.e. sum(x_i*y_i) for all i in vector)
z = data_bias.dot(weight)

def sigmoid(z):
	return 1/(1+np.exp(-z))


print(sigmoid(z))