import numpy as np

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def softmax(z):
	t = np.exp(z) # m * 10
	return t / np.sum(t, axis = 1, keepdims = True)
