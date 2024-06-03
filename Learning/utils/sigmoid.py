#Liangyz
#2024/6/3  下午2:49

import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z):
    return sigmoid(z) * (1 - sigmoid(z))


def sigmoid_derivative2(z):
    return np.exp(-z) / (1 + np.exp(-z))**2


def sigmoid_derivative3(z):
    return sigmoid(z) * (1 - sigmoid(z))


def sigmoid_derivative4(z):
    return sigmoid(z) * (1 - sigmoid(z))

