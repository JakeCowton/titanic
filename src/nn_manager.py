# Gives access to the NN

import numpy as np
from nn import NN

def _train(net, data, epochs, lr, momentum):
	"""
	Train `net` with `data`
	:param net: the network to train
	:param data: the data to train with
	:param epochs: The number full iterations through data
	:param lr: learning rate
	:param momentum: NN momentum
	"""
	for i in range(epochs):
		# Run one epoch
		for n in range(len(data)):
			net.feed_forward(data['inputs'][n])
			net.back_propagate(data['outputs'][n], lr, momentum)

def create_nn(data, structure, epochs=100, lr=0.01, momentum=0.1):
	"""
	Create and train the neural network
	:type data: ndarray
	:param data: Training data for the network
	:type structure: tuple
	:param structure: In, hidden and out layers (e.g. (5,4,3))
	:returns: A trained NN object
	"""
	ann = NN(structure)
	_train(ann, data, epochs, lr, momentum)
	return ann

def call_nn(nn, data):
	"""
	:type nn: NN object
	:param nn: The neural network
	:type data: array
	:param data: The input vars for the network
	:rtype: array
	:returns: The output layer
	"""
	return nn.feed_forward(data)

def find_mse(nn, data):

	mse = 0

	for i in range(data.size):
		given_out = nn.feed_forward(data['inputs'][i])
		expected_out = data['outputs'][i]

		diff = abs(expected_out - given_out)
		iter_error = 0
		for error in diff:
			iter_error += error**2

		iter_error = iter_error/len(diff)

		mse += iter_error

	mse = mse / len(data)

	return mse
