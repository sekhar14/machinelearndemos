import numpy as np


class perceptron(object):
	def __init__(self,eta=0.01,n_iter=10):
		self.eta = eta
		self.n_iter - n_iter


	def fit(self,X,y):
		self.w = np.zeros(X.shape[1] + 1)   

		for _ in range(self.n_iter):
			for xi,target in zip(X,y):
				error = (target - np.predict(xi))
				self.w[1:] += self.eta * error * xi
				self.w[0] += self.eta * error


	def net_input(self,X):
		return (np.dot(X,self.w[1:]) + self.w[0])

	def predict(self,X):
		np.where(self.net_input(X) >= 0, 1, -1)