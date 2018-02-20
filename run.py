import numpy as np
from matplotlib import pyplot as plt
from neural_net import neural_net
from IPython import embed

# Make the training set
A1 = 7 + np.random.randn(100)
A2 = 3 + np.random.randn(100)
A = np.matrix([list(a) + [1, 0, 0] for a in zip(A1, A2)])

B1 = 2 + np.random.randn(100)
B2 = 8 + np.random.randn(100)
B = np.matrix([list(b) + [0, 1, 0] for b in zip(B1, B2)])

C1 = 2 + np.random.randn(100)
C2 = 1 + np.random.randn(100)
C = np.matrix([list(c) + [0, 0, 1] for c in zip(C1, C2)])

D = np.concatenate((A, B, C))
np.random.shuffle(D)
X = D[:, :2] 
y = D[:, 2:]

# Normalize data
for i in range(2):
	mu = np.mean(X[:, i])
	sigma = np.std(X[:, i])
	X[:, i] = (X[:, i] - mu)/sigma

#z = y * np.matrix([[0],[1],[2]])
#plt.scatter(X[:, 0].A1, X[:, 1].A1, c = ((1/2)*z).A1)
#plt.show()

N = neural_net([2,4,4,3])
print('You have a neural_net object N.')

embed()
