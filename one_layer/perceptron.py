import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import time

# Define the perceptron class here.
class Perceptron(object):
	def __init__(self, learning_rate, training_set, targets, num_loops = 1):
		self.learning_rate = learning_rate
		self.num_loops = num_loops
		self.training_set = training_set
		self.targets = targets
		self.weights = np.zeros(1 + training_set.shape[1])
		#self.errors = []

	def fit(self):
		for _ in range(self.num_loops):
			#num_errors = 0
			n = 0
			while not self.overfit():
				n += 1
				for x, y in zip(self.training_set, self.targets):
					if self.predict(x) == y:
						continue
					update = self.learning_rate * (y - self.predict(x)) 

					# This could be improved by calculating the component of x normal to w and adding that instead.
					# This would also take care of the need for a learning rate, since points close the the boundary 
					# would correct the boundary by a small amount.
					self.weights[1:] += np.multiply(update, x) 
					self.weights[0] += update
					#if update != 0.0:
					#	num_errors += 1
					P.plot(True)
				#self.errors.append(num_errors)
				if n >= 25:
					print 'You may want to tweak the learning rate.'
					break
			print 'The perceptron has graduated.\nThe line is ' + self.learned_line() + \
			'.\nHopefully this is correct.'
		self.plot()

	def linear_input(self, x):
		return np.dot(x, self.weights[1:]) + self.weights[0]

	def predict(self, x):
		return np.where(self.linear_input(x) >= 0.0, 1, -1)

	# Visualization.  Taken from old script... I don't remember how this works.
	def _plot_decision_regions(self, resolution = 0.02):
		X = self.training_set
		y = self.targets
		markers = ('s', 'x', 'o', '^', 'v')
		colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
		cmap = ListedColormap(colors[:len(np.unique(y))])

		x1_min, x1_max = int(X[:, 0].min()), int(X[:, 0].max()) + 1
		x2_min, x2_max = int(X[:, 1].min()), int(X[:, 1].max()) + 1
		xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
		Z = self.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
		Z = Z.reshape(xx1.shape)
		plt.contourf(xx1, xx2, Z, alpha = 0.4, cmap = cmap)
		plt.xlim(xx1.min(), xx1.max())
		plt.ylim(xx2.min(), xx2.max())

		for idx, cl in enumerate(np.unique(y)):
			plt.scatter(x = X[y == cl, 0], y = X[y == cl, 1], alpha = 0.8, c = cmap(idx), marker = markers[idx], label = cl)

	# Plot shit.
	def plot(self, is_fitting = False):
		self._plot_decision_regions()
		plt.xlabel('some units')
		plt.ylabel('other units')
		plt.legend(loc = 'upper left')
		plt.show(block = False)
		if is_fitting:
			time.sleep(0.5)
			plt.close('all')

	def overfit(self):
		return all(self.predict(self.training_set) == self.targets)

	def learned_line(self):
		m = str(-self.weights[1] / self.weights[2])
		b = str(-self.weights[0] / self.weights[2])
		return 'y = ' + m + 'x + ' + b

# Training set.
#A1 = 7 + 2*np.random.randn(10)
#A2 = 3 + np.random.randn(10)
#A = np.array([list(a) + [1] for a in zip(A1, A2)])
#B1 = 2 + 3*np.random.randn(10)
#B2 = 8 + 2*np.random.randn(10)
#B = np.array([list(b) + [-1] for b in zip(B1, B2)])
#C = np.concatenate((A,B))
#np.random.shuffle(C)
#X = C[:, [0, 1]]
#y = C[:, [2]]

# For now, just use the full range for training data, since we plot the decision regions anyway.
#X_train = X
#y_train = y.transpose()[0]
X = 20*np.random.rand(50,2) - 10 
y = (X[:, 1] + X[:, 0]) > 1 # boundary to be learned is y = -x + 1 (chosen at random).

X_train = X
y_train = np.array(map(lambda x: 2*int(x) - 1, y))

# Get parameters from user.
learning_rate = input('Learning Rate: ')

# Initialize the perceptron.
P = Perceptron(learning_rate, X_train, y_train)

print 'You have a perceptron P.  Try entering \'P.fit()\'.'
