import numpy as np
from IPython import embed
from time import time

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
			W = np.append(W, X.A1)
		return W

	def unflatten(self, l, weights = []):
		# Unflattens weights to return weight matrix for layer l
		# l must be in range(1, len(nodes))
		if len(weights) == 0:
			weights = self.weights
		N = sum([self.nodes[i+1] * (self.nodes[i] + 1) for i in range(l-1)])
		return np.matrix(np.reshape(weights[N:N + self.nodes[l]*(self.nodes[l-1]+1)], (self.nodes[l], self.nodes[l-1]+1)))

	def cost(self, X, y, weights = []):
		# Returns the cost function J(theta) at X and y.
		# X and y are some kind of training, test, or validation set.
		if len(weights) == 0:
			weights = self.weights

		# y has shape m by K
		# h has shape m by K
		L = len(self.nodes) # number of layers (including input and output layers)
		K = self.nodes[L-1] # dimension of output layer (number of classes)
		m = X.shape[0] # number of training examples
		h = self.h(X, weights) # output matrix

		acc1 = 0
		for i in range(m):
			acc1 -= (y[i]*np.log(h[i].T) + (1-y[i])*np.log(1-h[i].T))[0,0]

		acc2 = 0
		for l in range(1, L):
			theta_squared = np.square(self.unflatten(l, weights)[:, 1:])
			acc2 += np.sum(theta_squared.A1)

		return (1/m) * (acc1 + (self.penalty/2)*acc2)

	def cost_gradient(self, X, y, weights = []):
		# Uses backprop algorithm from Andrew Ng's ML Course.
		# Want to return grad(weights) at the point (X, y, weights).
		if len(weights) == 0:
			weights = self.weights

		num_weights = weights.size
		L = len(self.nodes)
		Grad = [np.matrix(np.zeros([self.nodes[l], self.nodes[l-1]+1])) for l in range(1, L)]
		m = X.shape[0]

		for t in range(m):
			# Get all a's and z's.
			A, Z = self.forward_prop(X[t])
			err = []
			err.append(A[-1] - y[t].T) # append delta_L

			# The indexing here is very confusing.  What I want is err = [del_2, ..., del_L]
			# and Grad = [DEL_1, ..., DEL_{L-1}], according to Ng's notation (which
			# seems pretty standard).  Thus when I want to calculate DEL_l += a_l*del_{l+1},
			# indexing forces me to compute Grad[i] = Grad[i] + A[i]*err[i]
			for l in range(L-1, 1, -1):
				theta = self.unflatten(l, weights)
				delta_lplus1 = err[0] # always take the first element of this list
				gprime = self.insert_ones(np.multiply(A[l-1], 1-A[l-1]))
				delta_l = np.multiply(theta.T * delta_lplus1, gprime)
				err.insert(0, delta_l[1:]) # Stick the truncated delta at the beginning of err

			for i in range(L-1):
				a = self.insert_ones(A[i])
				Grad[i] = Grad[i] + err[i] * a.T

		Grad = [(1/m)*G for G in Grad] # Scale all gradients appropriately

		for i in range(L-1):
			theta = self.unflatten(i+1) # layer l indexed from one
			col = np.matrix(np.zeros([theta.shape[0], 1])) # column of zeros
			theta = np.append(col, theta[:, 1:], axis=1) # replace first column with zeros
			Grad[i] = Grad[i] + (self.penalty / m) * theta
		return self.flatten(Grad)

	#def numerical_gradient(self, e, X, y):
	#	# For debugging, this is a numerical approximation cost_gradient.  Hopefully the difference will be small.
	#	numgrad = np.zeros(self.weights.shape)
	#	perturb = np.zeros(self.weights.shape)
	#	for i in range(self.weights.size):
	#		perturb[i] = e
	#		loss1 = self.cost(X, y, self.weights + perturb)
	#		loss2 = self.cost(X, y, self.weights - perturb)
	#		numgrad[i] = (loss1 - loss2)/(2*e)
	#		perturb[i] = 0
	#	return numgrad

	def train(self, X, y):
		#costs = np.array([self.cost(X, y)])
		for i in range(50):
			self.weights -= self.learning_rate * self.cost_gradient(X, y)
			#costs = np.append(costs, self.cost(X, y))
		return self.cost(X, y)

	def sigmoid(self, z):
		exp = np.exp(-z)
		return 1/(exp + 1)

	#def sig_gradient(self, z):
	#	S = self.sigmoid(z)
	#	T = 1 - S # elementwise operation
	#	return np.multiply(S, T) # Hadamaard product of S and T

	def h(self, x, weights = []):
		if len(weights) == 0:
			weights = self.weights
		L = len(self.nodes) # number of layers (including input and output layers)
		a = x.T # a^(1) is x, which is a column vector
		for l in range(1, L):
			a = self.insert_ones(a) # stick a 1 in there
			theta = self.unflatten(l, weights)
			z = theta*a # z^(l+1) = Theta^(l)a^(l)
			a = self.sigmoid(z) # a^(l+1) = sigma(z^(l+1))
		return a.T # TODO: Should I return a or a.T?

	def predict(self, x, weights = []):
		# TODO: For pdr() method in run.py.  Needs to be generalized or deleted.
		z = self.h(x, weights)
		z2 = np.matrix(np.zeros(z.shape))
		M, N = z2.shape
		for i in range(M):
			for j in range(N):
				z2[i,j] = np.round(z[i,j])
		col = np.matrix([[0], [1], [2]])
		z3 = z2*col
		#return np.array(z3.flatten())[0]

		maxes = np.max(z, 1)
		preds = np.floor(z/maxes)
		z4 = preds * col
		return np.array(z4.flatten())[0]

	def forward_prop(self, x, weights = []):
		if len(weights) == 0:
			weights = self.weights
		L = len(self.nodes) # number of layers (including input and output layers)
		A = []
		Z = []
		a = x.T # a^(1) is x.T, which we think of as a row of column vectors, and we operate elementwise
		# (so just pretend x has dims self.nodes[0] x 1)
		A.append(a)
		for l in range(1, L):
			a = self.insert_ones(a) # stick a 1 in there
			theta = self.unflatten(l, weights)
			z = theta*a # z^(l+1) = Theta^(l)a^(l)
			Z.append(z)
			a = self.sigmoid(z) # a^(l+1) = sigma(z^(l+1))
			A.append(a)
		return A, Z

	def accuracy(self, X, y):
		# TODO: This also doesn't generalize.
		h = self.h(X)
		maxes = np.max(h, 1) # column vector of maxes of each row
		preds = np.floor(h/maxes) # predictions of the form [i, j, k], where exactly one of i,j,k is 1, and the other two are 0
		correct = np.all(preds == y, 1)
		return np.sum(correct.A1) / y.shape[0]

	def plot_decision_regions(self):
		# TODO
		print('under construction.')
