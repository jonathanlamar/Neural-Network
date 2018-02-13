import numpy as np

class neural_net(object):
	def __init__(self, nodes_vector = [2,1]):
		# Store the architecture
		self.nodes = nodes_vector

		# Weights initialized to random (and flat).
		N = sum([self.nodes[i+1] * (self.nodes[i] + 1) for i in range(len(self.nodes)-1)])
		self.weights = np.random.randn(N)

		# Learning rate.  Used to scale cost gradient vector for backpropagation.
		self.learning_rate = 0.1

		# Usually written as lambda.  penalty factor for the norm of the matrix
		self.penalty = 1

	def set_learning_rate(self, val):
		self.learning_rate = val

	def set_penalty(self, val):
		self.penalty = val

	def theta(self, l, weights = self.weights):
		N = sum([self.nodes[i+1] * (self.nodes[i] + 1) for i in range(l-1)])
		return np.matrix(np.reshape(self.weights[N:N + self.nodes[l]*(self.nodes[l-1]+1)], (self.nodes[l], self.nodes[l-1]+1)))

	def cost(self, X, y, weights = self.weights):
		# y has shape m by K
		# h has shape m by K
		L = len(self.nodes) # number of layers (including input and output layers)
		K = self.nodes[-1] # dimension of output layer (number of classes)
		m = X.shape[0] # number of training examples
		h = self.h(X) # output matrix

		acc = 0
		for i in range(1, m+1):
			acc -= (y[i].transpose()*np.log(h[i]) + (1-y[i]).transpose()*np.log(1-h[i]))

		acc2 = 0
		for l in range(L):
			acc2 += sum(sum(self.theta(l, weights)[:, 1:]**2))

		return (1/m) * (acc + (self.penalty/2)*acc2)

	def cost_gradient(self, i):
		# Uses backprop algorithm from Andrew Ng's ML Course.
		# TODO
		print('Under construction.')

	def numerical_gradient(self, e, X, y):
		numgrad = np.zeros(self.weights.shape)
		perturb = np.zeros(self.weights.shape)
		for i in range(self.weights.size):
			perturb[i] = e
			loss1 = self.cost(X, y, self.weights - perturb)
			loss2 = self.cost(X, y, self.weights + perturb)
			numgrad[i] = (loss1 - loss2)/(2*e)
			perturb[i] = 0
		return numgrad

	def train(self, X):
		# TODO
		print('Under construction.')

	def sigmoid(self, z):
		exp = np.exp(-z)
		return (exp + 1)**(-1)

	def h(self, x):
		L = len(self.nodes) # number of layers (including input and output layers)
		a = x # a^(1) is x
		for l in range(1, L):
			a = np.insert(a, 0, 1) # stick a 1 in there
			z = NN.theta(l)*a # z^(l+1) = Theta^(l)a^(l)
			a = self.sigmoid(z) # a^(l+1) = sigma(z^(l+1))
		return a
