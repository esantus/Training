# This code was written by Enrico Santus
#
# Suppose you have some data recording info about house size, x, and
# price, y. 
# You want to create a model that predicts the price of a
# house given its size.
# Let's see how to do it.

import numpy as np
import matplotlib.pyplot as plt
from linearRegressionLib import *


# Let's create a vector with N dimensions in range [0,1]
N = 5
house_size = np.random.rand(1, N)

# Let's add 0.5 to each value and multiply them by 60 times to have
# some likely dimensions.
for i, record in enumerate(house_size):
	house_size[i] = (record+0.5)*60

# Let's store how many records we have in our dataset (we could use
# the constant N, but we want to learn how to get the shape of our
# data)
_, dim_y = house_size.shape

# Let's create some fake training prices, which are linearly related
# to the house sizes. In this way, later, we have to discover the
# parameters m and q of the function: y = mx + q
set_m = 0
set_q= -1.5
house_price_training = np.empty([1, dim_y])
for i, record in enumerate(house_size):
	house_price_training[i] = (record*set_m)+set_q

# Let's print our sizes and prices
print("House size: " + str(house_size))
print("House price: " + str(house_price_training))
print("House prices were calculated with the following function: y = ", str(set_m), "x + ", str(set_q))
# Let's start modelling and finding the parameters that can be used to
# predict new prices, given the house size.
found_m, found_q = linearRegression(house_size, house_price_training)

# Let's print what our linearRegression function found and how we can predict
# prices in the future.
print("m = " + str(found_m))
print("q = " + str(found_q))
print("House prices can be predicted with the following function: y = ", str(found_m), "x + ", str(found_q))

# Let's print the shape of our vectors
print (house_size.shape, house_price_training.shape, fit_fn(house_size, found_m, found_q).shape)

# Let's create a list of house size for which the price is unknown
N = 5
house_size_new = np.random.rand(1, N*N)

# Let's add 0.5 to each value and multiply them by 60 times to have
# some likely dimensions.
for i, record in enumerate(house_size_new):
	house_size_new[i] = (record+0.5)*60

# Let's plot our results
plt.plot(house_size_new, fit_fn(house_size_new, found_m, found_q), 'yo')
plt.show()
