# This code is an adaptation of the code published in:
# https://github.com/lazyprogrammer/machine_learning_examples/

import numpy as np
import pandas as pd


# Load the data
def get_data():

	#Data in six columns: is_mobile,n_products_viewed,visit_duration,is_returning_visitor,time_of_day,user_action (e.g. 1,0,0.657509946224,0,3,0)
	print("Loading the CSV")
	df = pd.read_csv('ecommerce_data.csv')

	#print("Printing the head of the dataframe:")
	#print(df.head())

	# Turning data into a np matrix
	data = df.as_matrix()

	#print("Printing the head of the np matrix")
	#print(data)

	# Everything but the Prediction
	X = data[:, :-1]

	# Predictions
	Y = data[:, -1]

	# Normalize columns 1 and 2 (non categorical)
	X[:,1] = (X[:,1]-X[:,1].mean()) / X[:,1].std()
	X[:,2] = (X[:,2]-X[:,2].mean()) / X[:,2].std()

	# Save the dimensions of the np matrix
	ITEMS, DIMENSIONS = X.shape

	# Create another matrix which contains three extra columns for rapresenting the categorical variable (0 < time_of_the_day < 3) in four binary dimensions.
	X2 = np.zeros((ITEMS, DIMENSIONS+3))

	# Initialize part of this matrix with X
	X2[:,0:(DIMENSIONS-1)] = X[:,0:(DIMENSIONS-1)]

	# Create three extra columns to represent the time of the day in binary way
	for n in range(ITEMS):
		# Time of the day: 0 < t < 3
		t = int(X[n, DIMENSIONS-1])
		X2[n, t+DIMENSIONS-1] = 1

	# Another method to create the binary elements
	#Z = np.zeros((ITEMS, 4))
	#Z[np.arange(ITEMS), X[:,DIMENSIONS-1].astype(np.int32)] = 1
	#X2[:,-4:] = Z
	#assert(np.abs(X2[:,-4:]-Z).sum() < 10e-10)

	return X2, Y


# Return only binary data
def get_binary_data():

	X, Y = get_data()

	# Save all the items where Y is equal to 0 or 1
	X2 = X[Y <= 1]
	Y2 = Y[Y <= 1]

	return X2, Y2




