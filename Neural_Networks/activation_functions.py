import numpy as np

def sigmoid(x):
    x = np.clip(x, -500, 500) 
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1 - s)


def tanh(x):
    return np.tanh(x)

def tanh_derivative(z):
    t = np.tanh(z)
    return 1.0 - t**2


def relu(x):
    return np.maximum(0, x)

def relu_derivative(z):
    return np.where(z > 0, 1.0, 0.0)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)