import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from time import time

class neural_net(object):
	def __init__(self, nodes_vector):
		# Store the architecture
		self.nodes = nodes_vector
		
		# Weights and biases.  Initialized to random.
		# TODO: Unroll weights and roll back when needed for matrix multiplication
		self.weights = []
		for i in range(len(nodes_vector)-1):
			self.weights.append(np.matrix(np.random.randn(nodes_vector[i+1], nodes_vector[i])))
		
		# Learning rate.  Used to scale cost gradient vector for backpropagation.
		self.learning_rate = 0.1

		# Usually written as lambda.  penalty factor for the norm of the matrix
		self.penalty = 0

	def set_learning_rate(self, val):
		self.learning_rate = val

	def set_penalty(self, val):
		self.penalty = val

	def cost(self, X, y):
		# y has shape m by K
		# h has shape m by K
		L = len(self.nodes) # number of layers (including input and output layers)
		K = self.nodes[-1] # dimension of output layer (number of classes)
		m = X.shape[0] # number of training examples
		h = self.h(X) # output matrix

		# TODO Vectorize this later.
		acc = 0
		for i in range(1, m+1):
			acc -= (1/m)*(y[i]*np.log(h[i]).transpose() + (1-y[i])*np.log(1-h[i]).transpose())

		# TODO Vectorize this later.
		for l in range(L-1):
			for i in range(1, self.nodes[l+1]):
				for j in range(1, self.nodes[l]):
					acc += (self.penalty / (2*m)) * self.weights[l][i,j]**2

		return acc

	def cost_gradient(self, i):
		# Uses backprop algorithm from Andrew Ng's ML Course.
		# TODO
		print('Under construction.')

	def train(self, X):
		# TODO
		print('Under construction.')

	def backprop(self, batch):
		# TODO
		print('Under construction.')

	def sigmoid(self, z):
		exp = np.exp(-z)
		return (exp + 1)**(-1)

	def h(self, x):
		z1 = self.weights[0] * x + self.biases[0]
		a1 = self.sigmoid(z1)
		z2 = self.weights[1] * a1 + self.biases[1]
		return self.sigmoid(z2)

	def classify(self, t):
		# TODO
		print('Under construction.')

	def plot(self):
		# TODO: This doesn't work
		print('Under construction.')
