import numpy as np


def relu(output):
    return np.maximum(output, 0)


def softmax(output):
    #print(output)
    total_exp = np.sum(np.exp(output), axis=0)
    results = np.exp(output)/total_exp
    return results


def tanh(output):
    return np.tanh(output)


def relu_deriv(output):
    return np.array(output > 0, dtype=np.int8)


def tanh_deriv(output):
    return 1 - np.tanh(output)**2


def loss_function(true_value, predicted_value):
    return - np.sum(np.multiply(true_value, np.log(predicted_value)))/predicted_value.shape[-1]


def error_function(true_value, predicted_value):
    return np.sum(np.multiply(true_value,(predicted_value)))/predicted_value.shape[-1]

