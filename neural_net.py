import numpy as np
from ipython import embed()

class neural_net(object):
	def __init__(self, nodes_vector = [2,1]):
		# Store the architecture
		self.nodes = nodes_vector

		# Weights initialized to random (and flat).
		N = sum([self.nodes[i+1] * (self.nodes[i] + 1) for i in range(len(self.nodes)-1)])
		self.weights = np.random.randn(N) # initialize weights

		# Learning rate.  Used to scale cost gradient vector for backpropagation.
		self.learning_rate = 0.1

		# Usually written as lambda.  penalty factor for the norm of the matrix
		self.penalty = 1

	def set_learning_rate(self, val):
		self.learning_rate = val

	def set_penalty(self, val):
		self.penalty = val

	def insert_ones(self, X):
		# inserts a row of ones
		row = np.ones([1, X.shape[1]])
		return np.append(row, X, axis=0)

	def flatten(self, matrices):
		# Flattens a list of matrices into an array.
		# No check on dimension, although it should match
		# self.nodes and self.weights.
		W = np.array([])
		for X in matrices:
			W = np.append(W, np.reshape(np.array(X), X.size))
		return W

	def unflatten(self, l, weights = self.weights):
		# Unflattens weights to return weight matrix for layer l
		# l must be in range(1, len(nodes))
		N = sum([self.nodes[i+1] * (self.nodes[i] + 1) for i in range(l-1)])
		return np.matrix(np.reshape(self.weights[N:N + self.nodes[l]*(self.nodes[l-1]+1)], (self.nodes[l], self.nodes[l-1]+1)))

	def cost(self, X, y, weights = self.weights):
		# Returns the cost function J(theta) at X and y.
		# X and y are some kind of training, test, or validation set.

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

	def cost_gradient(self, X, y, weights = self.weights):
		# Uses backprop algorithm from Andrew Ng's ML Course.
		# Want to return grad(weights) at the point (X, y, weights).
		J = self.cost(X, y, weights)
		num_weights = weights.size
		L = len(self.nodes)
		Grad = [np.matrix(np.zeros([self.nodes[l], self.nodes[l-1]+1])) for l in range(1, L)]
		m = X.shape[1]

		for t in range(m):
			# Get all a's and z's.
			A = self.forward_prop(X[t].transpose())
			err = []
			err.append(A[-1] - y[t].transpose()) # append delta_L

			# The indexing here is very confusing.  What I want is err = [del_2, ..., del_L]
			# and Grad = [DEL_1, ..., DEL_{L-1}], according to Ng's notation (which
			# seems pretty standard).  Thus when I want to calculate DEL_l += a_l*del_{l+1},
			# indexing forces me to compute Grad[i] = Grad[i] + A[i]*err[i]
			for l in range(1, L):
				l = i + 1
				delta = np.multiply(self.theta(L-(i+1)).transpose()*ERR[-(i+1)], self.sig_gradient())
				err.insert(0, delta[1:]) # Stick the truncated delta at the beginning of err

			for i in range(L-1):
				Grad[i] = Grad[i] + err[i] * A[i].transpose()

		Grad = [(1/m)*G for G in Grad] # Scale all gradients appropriately

		for i in range(L-1):
			theta = self.unflatten(i+1) # layer l indexed from one
			col = np.matrix(np.zeros([theta.shape[0], 1])) # column of zeros
			theta = np.append(col, theta[:, 1:], axis=1) # replace first column with zeros
			Grad[i] = Grad[i] + (self.penalty / m) * theta

	def numerical_gradient(self, e, X, y):
		# For debugging, this is a numerical approximation cost_gradient.  Hopefully the difference will be small.
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

	def sig_gradient(self, z):
		S = self.sigmoid(z)
		T = 1 - S # elementwise operation
		return np.multiply(S, T) # Hadamaard product of S and T

	def h(self, x):
		L = len(self.nodes) # number of layers (including input and output layers)
		a = x # a^(1) is x, which is a column vector
		for l in range(1, L):
			a = np.insert(a, 0, 1) # stick a 1 in there
			z = NN.theta(l)*a # z^(l+1) = Theta^(l)a^(l)
			a = self.sigmoid(z) # a^(l+1) = sigma(z^(l+1))
		return a

	def forward_prop(self, x):
		L = len(self.nodes) # number of layers (including input and output layers)
		A = []
		a = x # a^(1) is x, which is a column vector
		A.append(a)
		for l in range(1, L):
			a = np.insert(a, 0, 1) # stick a 1 in there
			z = NN.theta(l)*a # z^(l+1) = Theta^(l)a^(l)
			a = self.sigmoid(z) # a^(l+1) = sigma(z^(l+1))
			A.append(a)
		return A
