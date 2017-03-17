__author__ = 'Sekhar Banarjee'


import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt


np.random.seed(0)
X,y = make_moons(200,noise=0.20)
plt.scatter(X[:,0],X[:,1],c = y, cmap = plt.cm.Spectral)
plt.show()