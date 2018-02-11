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

	def theta(self, l):
		N = sum([self.nodes[i+1] * (self.nodes[i] + 1) for i in range(l-1)])
		return np.matrix(np.reshape(self.weights[N:N + self.nodes[l]*(self.nodes[l-1]+1)], (self.nodes[l], self.nodes[l-1]+1)))

	def debug_cost(self, X, y):
		# TODO: Remove if this matches self.cost.
		# y has shape m by K
		# h has shape m by K
		L = len(self.nodes) # number of layers (including input and output layers)
		K = self.nodes[-1] # dimension of output layer (number of classes)
		m = X.shape[0] # number of training examples
		h = self.h(X) # output matrix
		logh = np.log(h)
		one_minus_y = np.ones(y.shape) - y
		log_one_minus_h = np.log(np.ones(h.shape) - h)

		acc = 0
		for i in range(1, m+1):
			for k in range(1, K+1):
				acc -= (y[i,k]*logh[i,k] + one_minus_y[i,k]*log_one_minus_h[i,k])

		acc2 = 0
		for l in range(self.weights.shape[0]):
			acc2 += self.weights[l]**2

		return (1 / m) * (acc + self.penalty * acc2 / 2)

	def cost(self, X, y):
		# y has shape m by K
		# h has shape m by K
		L = len(self.nodes) # number of layers (including input and output layers)
		K = self.nodes[-1] # dimension of output layer (number of classes)
		m = X.shape[0] # number of training examples
		h = self.h(X) # output matrix

		acc = 0
		for i in range(1, m+1):
			acc -= y[i]*np.log(h[i]).transpose() + (1-y[i])*np.log(1-h[i]).transpose()

		return (1 / m) * (acc + self.penalty * sum(self.weights**2) / 2)

	def cost_gradient(self, i):
		# Uses backprop algorithm from Andrew Ng's ML Course.
		# TODO
		print('Under construction.')

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
