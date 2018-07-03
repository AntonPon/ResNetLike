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