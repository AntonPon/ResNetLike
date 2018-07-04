import numpy as np


def relu(output):
    return np.maximum(output, 0)


def softmax(output):
    total_exp = np.sum(np.exp(output))
    results = np.zeros(output.size[0])
    results[:] = np.exp(output)/total_exp
    return results


def tanh(output):
    return np.tanh(output)


def relu_deriv(output):
    result = np.zeros(output.size(0))
    for i in enumerate(result):
        if output[i] > 0:
            result[i] = 1
    return result


def tanh_deriv(output):
    return (1 - np.tanh(output)**2)

def error_function(true_value, predicted_value):
    return - np.sum(np.multiply(true_value, np.log(predicted_value)))


