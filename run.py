execfile('neural_net.py')
from sklearn import datasets
iris = datasets.load_iris()

X= iris.data[:, [2, 3]]
T = iris.target

# Plot stuff.
markers = ('s', 'x', 'o', '^', 'v')
colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
cmap = ListedColormap(colors[:len(np.unique(T))])
x1_min, x1_max = X[:, 0].min(), X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min(), X[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.02), np.arange(x2_min, x2_max, 0.02))
plt.xlim(xx1.min(), xx1.max())
plt.ylim(xx2.min(), xx2.max())
for idx, cl in enumerate(np.unique(T)):
	plt.scatter(x = X[T == cl, 0], y = X[T == cl, 1], alpha = 0.8, c = cmap(idx), marker = markers[idx], label = cl)
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc = 'upper left')
plt.show(block = False)

# Pre-process X and T into the correct form.
Xprime = [np.matrix([[x[0]], [x[1]]]) for x in X]
Tprime = []
for t in T:
	if t == 0:
		Tprime.append(np.matrix([[1], [0], [0]]))
	elif t == 1:
		Tprime.append(np.matrix([[0], [1], [0]]))
	elif t == 2:
		Tprime.append(np.matrix([[0], [0], [1]]))

# Initialize the perceptron.
P = multilayer_perceptron(Xprime, Tprime)

print('You have a neural network.  Its name is P.  Its training set is the iris petal and sepal \
length dataset.  Try its methods out.')
