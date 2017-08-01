def mean(x):
	"""
	This function calculates the mean value of a list of numbers.
	It is equivalent to writing x.mean()
	Args:
		x (np array): array
	Returns:
		float: mean (as x.mean())
	"""
	return x.sum() / float(x.shape[1])


def variance(x, mean):
	"""
	This function calculates the variance of the values in x
	It is equivalent to writing x.var()
	Args:
		x (np array): array
		mean (float): mean of x array
	Returns:
		float: variance (as x.var())
	"""
	
	#The generator creates a list of np arrays, so we select the first one
	return [(i-mean)**2 for i in x][0].sum()


def covariance(x, mean_x, y, mean_y):
	"""
	This function calculates the co-variance of x and y
	Args:
		x (np array): array
		mean_x (float): mean of x array
		y (np array): array
		mean_y (float): mean of y array
	Returns:
		float: co-variance
	"""
	
	covar = 0.0
	for i in range(x.shape[1]):
		covar += (x[0][i]-mean_x) * (y[0][i]-mean_y)
	return covar


def linearRegression(x, y):
	"""
	This function takes in input the np array containing the x values
	and the array containing the y values and return the parameters m
	and q of the function y = mx + q which model the distribution.
	"""

	# Let's calcualte the mean and variance of x and y
	mean_x, mean_y = mean(x), mean(y) #x.mean(), y.mean()
	var_x, var_y = variance(x, mean_x), variance(y, mean_y) #x.var(), y.var()


	# Let's calculate the co-variance of x and y
	covar = covariance(x, mean_x, y, mean_y)

	# Now we can calculate the coefficients m and q of the line that
	# connects the various points in our dataset: y = mx + q
	m = covar / var_x # Co-variance / Variance of x
	q = mean_y - m * mean_x # y = mx + q --> q = y - mx

	return m, q


# This function can be used for predictions
def fit_fn(x, m, q):
	return [(m*i)+q for i in x][0].reshape((x.shape)) # The generator generates an array of np.arrays, so we only take the first
