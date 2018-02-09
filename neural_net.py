import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from time import time

class neural_net(object):
	def __init__(self, training_set, targets, learning_rate = 0.1):
		r"""
		GOAL: To do a three-region classifier using a multilayer perceptron.
		I am guessing three layers might do it (like learning XOR) but I could be wrong.
		"""

		# Two-dimensional input data (i.e., two input neurons),
		# Three-dimensional hidden layer (i.e., three hidden neurons),
		# Three-dimensional output (i.e., three output neurons).

		if len(training_set) != len(targets):
			raise ValueError('training_set and targets need to have the same length.')
		if any([x.shape != (2, 1) for x in training_set]):
			raise ValueError('Elements of training_set must be numpy matrices of shape (2, 1).')
		if any([y.shape != (3, 1) for y in targets]):
			raise ValueError('Elements of target must be numpy matrices of shape (3, 1).')

		# Training set and desired outputs.
		self.training_set = training_set
		self.targets = targets

		# Weights and biases.  Initialized to random.
		self.weights = [np.matrix(np.random.randn(3, 2)), np.matrix(np.random.randn(3, 3))]
		
		# Ng's NN lesson does not have a bias vector at each step...
		#self.biases = [np.matrix(np.random.randn(3, 1)), np.matrix(np.random.randn(3, 1))]

		# Learning rate.  Used to scale cost gradient vector for backpropagation.
		self.learning_rate = learning_rate

		# Usually written as lambda.  penalty factor for the norm of the matrix
		self.penalty = 0

	def cost(self, prediction, target):
		r"""
		Cost (error, i.e., negative log likelihood) function.  prediction must be
		of the form self.predict(x), where x corresponds to the target.
		"""
		# Input will probably be numpy matrices or numpy arrays.  Output will be a numpy array.
		# About 28 microseconds on my laptop.
		y = np.array(prediction)
		t = np.array(target)
		return sum(np.square(y - t))[0]

	def overfit(self):
		r"""
		Returns True if the network correctly predicts all training data.
		"""
		return all([self.cost(self.predict(x), t) == 0 for x, t in zip(self.training_set, self.targets)])

	def cost_gradient(self, i):
		# TODO
		prtin('Under construction.')

	def train(self):
		# TODO
		prtin('Under construction.')

	def backprop(self, batch):
		# TODO
		prtin('Under construction.')

	def sigmoid(self, x):
		r"""
		Sigmoid (logistic) function: returns e^x/(1+e^x)
		"""
		exp = np.exp(x)
		return exp / (exp + 1)

	def h(self, x):
		z1 = self.weights[0] * x + self.biases[0]
		a1 = self.sigmoid(z1)
		z2 = self.weights[1] * a1 + self.biases[1]
		return self.sigmoid(z2)

	def classify(self, t):
		t_max = max(max(t[0, 0], t[1, 0]), t[2, 0])
		if t[0, 0] == t_max:
			return 0
		elif t[1, 0] == t_max:
			return 1
		else:
			return 2

	def plot(self):
		# TODO: This doesn't work
		X = self.training_set
		T = [self.classify(t) for t in self.targets]
		markers = ('s', 'x', 'o', '^', 'v')
		colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
		cmap = ListedColormap(colors[:len(np.unique(T))])

		x1_min, x1_max = X[:, 0].min(), X[:, 0].max() + 1
		x2_min, x2_max = X[:, 1].min(), X[:, 1].max() + 1
		xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02), np.arange(x2_min, x2_max, 0.02))
		Z = self.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
		Z = Z.reshape(xx1.shape)
		plt.contourf(xx1, xx2, Z, alpha = 0.4, cmap = cmap)
		plt.xlim(xx1.min(), xx1.max())
		plt.ylim(xx2.min(), xx2.max())

		for idx, cl in enumerate(np.unique(T)):
			plt.scatter(x = X[T == cl, 0], y = X[T == cl, 1], alpha = 0.8, c = cmap(idx), marker = markers[idx], label = cl)
