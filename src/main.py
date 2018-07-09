from ResNet import TheResNet
from functions import tanh, relu
from utls.data_prep import data_prep
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt

from utls.grid_search import AwesomeGridSearch


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
    net_tanh_out.fit(X.T, y)
    net_tanh_out.train()
    plot_results(net_tanh_out.test_err, 'Network with tanh on the output layer')

    # GridSearch Demo

    grs = AwesomeGridSearch(TheResNet, X=X.T, y=y)
    params = {
        'learning_rate': [0.01, 0.05, 0.1],
        'dim_hidden1': [5, 10, 20],
        'max_epochs': [100],
        'batch_size': [10,20,30]
    }

    best_params, best_score, worst_params, worst_score = grs.search(params)

    print('Best Score: {}'.format(best_score))
    print('Best Params: {}'.format(best_params))

    best_net = TheResNet(**best_params)
    best_net.fit(X.T, y)
    best_net.train()
    print('Best Net Precision on Test: {}'.format(best_net.test_err[-1]))
    print('Best Net Precision on Train: {}'.format(best_net.train_err[-1]))




