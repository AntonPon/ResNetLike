from ResNet import TheResNet
from functions import tanh, relu
from utls.data_prep import data_prep
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt


def plot_results(precision, title):
    plt.figure(figsize=(12, 8))
    plt.xlabel('epoch')
    plt.plot( range(1, len(precision) + 1, 1), precision)
    plt.ylabel('accuracy')
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    # reading data
    y, X = data_prep('../data/wdbc.data')
    m, n = X.shape
    X = normalize(X, axis=0)

    # just training with the default, tuned params params
    net1 = TheResNet()
    net1.fit(X.T, y)
    net1.train()
    plot_results(net1.test_err, 'Default Network Results')

    # replacing softmax with tanh

    net_tanh_out = TheResNet(act_funcs=[tanh, relu, tanh])
    net_tanh_out.fit(X, y)
    net_tanh_out.train()
    plot_results(net_tanh_out.test_err, 'Network with tanh on the output layer')

    # GridSearch Demo


