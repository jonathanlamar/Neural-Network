import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from time import time

class multilayer_perceptron(object):
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
		self.biases = [np.matrix(np.random.randn(3, 1)), np.matrix(np.random.randn(3, 1))]

		# Learning rate.  Used to scale cost gradient vector for backpropagation.
		self.learning_rate = learning_rate

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
		r"""
		Approximation of the gradient of self.cost as a function of the weights
		and biases at the point determined by prediction and target.
		"""
		# This returns a list with four numpy matrices: one for weights[0],
		# one for biases[0], one for weights[1], and one for biases[1].

		# First get relevant activations and whatnot.
		x = self.training_set[i]
		y = self.predict(x)
		t = self.targets[i]
		z1 = self.hidden_layer(x)
		a1 = self.sigmoid(z1)
		z2 = self.output_layer(x)
		a2 = self.sigmoid(z2)
		del_w1 = np.matrix(np.zeros([3, 2]))
		del_b1 = np.matrix(np.zeros([3, 1]))
		del_w2 = np.matrix(np.zeros([3, 3]))
		del_b2 = np.matrix(np.zeros([3, 1]))

		# TODO: DEBUG THE CRAP OUT OF IT
		# Compute del_w1.
		for i in [0, 1, 2]:
			for j in [0, 1]:
				for k in [0, 1, 2]:
					del_w1[i, j] += 2 * (a2[k] - t[k]) * \
					self.sigmoid_prime(z2[k]) * self.weights[1][k,i] * self.sigmoid_prime(z1[i]) * x[j]

		# Compute del_b1.
		for i in [0, 1, 2]:
			for j in [0, 1, 2]:
				del_b1[i, 0] += 2 * (a2[j] - t[j]) * \
				self.sigmoid_prime(z2[j]) * self.weights[1][j,i] * self.sigmoid_prime(z1[i])

		# Compute del_w2.
		for i in [0, 1, 2]:
			for j in [0, 1, 2]:
				del_w2[i, j] += 2 * (a2[i] - t[i]) * self.sigmoid_prime(z2[i]) * a1[j]

		# Compute del_b2.
		for i in [0, 1, 2]:
			for j in [0, 1, 2]:
				del_b1[i, 0] += 2 * (a2[i] - t[i]) * self.sigmoid_prime(z2[j])

		return del_w1, del_b1, del_w2, del_b2

	def train(self):
		r"""
		Main training algorithm.  Repeatedly uses backpropagation to get cost down.
		"""
		# TODO - Make batches (should we just use the whole training set?)
		# for batch in batches, self.backprop(batch), etc.
		# For now using the whole training set.
		n = 0
		while not self.overfit():
			n += 1
			self.backprop(range(len(self.training_set)))
			if n > 100:
				print 'Training algorithm doesn\'t seem to work...\n\
				You may want to tweak... any number of things.'
				return
		print 'Welp, the network is now overfit.'

	def backprop(self, batch):
		r"""
		Backpropagation algorithm.  Estimate cost gradient as function of weights
		and biases and subtract a scalar multiple of it 
		"""
		# This is probably super slow.
		batch_size_inv = np.divide(1., len(batch))
		# TODO - This will probably need to be changed.
		print 'doing backprop...'
		t = time()
		for i in batch:
			cost = self.cost(self.predict(self.training_set[i]), self.targets[i])
			del_w1, del_b1, del_w2, del_b2 = self.cost_gradient(i)
			self.weights[0] -= self.learning_rate * cost * batch_size_inv * del_w1
			self.biases[0] -= self.learning_rate * cost * batch_size_inv * del_b1
			self.weights[1] -= self.learning_rate * cost * batch_size_inv * del_w2
			self.biases[1] -= self.learning_rate * cost * batch_size_inv * del_b2
		print 'done in ' + str(time() - t) + ' seconds.'

	def sigmoid(self, x):
		r"""
		Sigmoid (logistic) function: returns e^x/(1+e^x)
		"""
		exp = np.exp(x)
		return exp / (exp + 1)

	def sigmoid_prime(self, x):
		r"""
		Derivative of sigmoid function
		"""
		return self.sigmoid(-x)

	def hidden_layer(self, x):
		r"""
		Returns the values of the hidden neurons at the point x
		"""
		return self.weights[0] * x + self.biases[0]

	def output_layer(self, x):
		r"""
		Returns the values of the output neurons at the point x
		"""
		a1 = self.sigmoid(self.hidden_layer(x))
		return self.weights[1] * a1 + self.biases[1]

	def predict(self, x):
		r"""
		The point of all this.  Guesses the class of x based on training experience.
		"""
		return self.sigmoid(self.hidden_layer(x))

	def classify(self, t):
		t_max = max(max(t[0, 0], t[1, 0]), t[2, 0])
		if t[0, 0] == t_max:
			return 0
		elif t[1, 0] == t_max:
			return 1
		else:
			return 2

	def plot(self):
		r"""
		For visualization.
		"""
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
