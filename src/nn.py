
import numpy as np

def sigmoid(x):
	"""
	Uses tanh() for a sigmoid function
	:param x: The number to pass into the sigmoid function
	:returns: tanh(x)
	"""
	return np.tanh(x)

def sigmoid_first_d(x):
	"""
	:param x: The outputs to pass into the sigmoid's first derivative function
	:returns: The first derivative of the sigmoid function on x
	"""
	return 1.0-x**2


class NN(object):
	""" Artificial Neural Network class """

	def __init__(self, structure):
		"""
		Initialises the ANN
		:type structure: tuple
		:param structure: number of input, hidden and output layers
		"""

		# Store the structure of the network
		self.structure = structure
		self.no_of_layers = len(structure)

		self.layers = []

		# Create the first layer (with added bias)
		self.layers.append(np.ones(self.structure[0]+1))

		# Create the rest of the layers
		for i in range(1,self.no_of_layers):
			self.layers.append(np.ones(self.structure[i]))

		# Generate weights matrix between 0 and 1
		self.weights = []
		for i in range(self.no_of_layers-1):
			self.weights.append(np.random.random_sample((
												self.layers[i].size,
                                         		self.layers[i+1].size)))

		# Difference in weights
		self.dw = [0,]*len(self.weights)

	def feed_forward(self, data):
		"""
		Feeds `data` through the network
		:type data: ndarray
		:param data: Input values
		:returns: Output layer
		"""

		# Set the input layer to equal data (-1 to keep the bias as 1)
		try:
			self.layers[0][0:-1] = data
		except TypeError:
			from ipdb import set_trace; set_trace()

		for i in range(1, self.no_of_layers):
			# Feed the data forward using `np.dot`
			# `sigmoid` is the activating function
			self.layers[i][...] = sigmoid(np.dot(self.layers[i-1],
											     self.weights[i-1]))

		# Return the last layer (output layers)
		return self.layers[-1]

	def back_propagate(self, target, lr=0.1, momentum=01):
		"""
		Back propage the network with a target value
		:type target: float
		:param target: The expected output
		:type lr: float
		:param lr: The network learning rate
		:type momentum: float
		:param momentum: The M value of the network
		:returns: Error in the network
		"""

		changes = []

		# Calculate output error
		self.error = target - self.layers[-1]
		change = self.error*sigmoid_first_d(self.layers[-1])
		changes.append(self.error)

		# Calculate hidden errors
		# This range is from the hightest layer down to the lowest
		for i in range(self.no_of_layers-2,0,-1):
			change = np.dot(changes[0],self.weights[i].T)*\
				sigmoid_first_d(self.layers[i])
			changes.insert(0, change)


		# Update weights
		for i in range(len(self.weights)):
			layer = np.atleast_2d(self.layers[i])
			diff = np.atleast_2d(changes[i])
			weight_difference = np.dot(layer.T, diff)
			self.weights[i] += lr*weight_difference + momentum*self.dw[i]
			self.dw[i] = weight_difference

		return (self.error**2).sum()
