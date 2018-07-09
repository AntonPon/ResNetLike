import numpy as np
from matplotlib import pyplot as plt

from src.functions import tanh, relu, softmax, relu_deriv, tanh_deriv, error_function
from sklearn.preprocessing import normalize

from utls.data_prep import data_prep


class TheResNet(object):
    def __init__(self, dim_in, dim_hidden1, dim_hidden2, dim_out, learning_rate=7*10**(-3), batch_size=10,
                 rand_seed=42, max_epochs=200,
                 act_funcs = (tanh, relu, softmax),
                 act_func_derivs  = (tanh_deriv, relu_deriv)):
        self.dim_in = dim_in
        self.dim_hidden1 = dim_hidden1
        self.dim_hidden2 = dim_hidden2
        self.dim_out = dim_out
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.rand_seed = rand_seed
        self._init_weights(init_type='xavier')
        self.X = None
        self.y = None
        self.m = None
        self.max_epoch = max_epochs
        self.epoch = 0
        self.train_err = []
        self.test_err = []
        self.act_funcs = act_funcs
        self.act_func_derivs = act_func_derivs

    def _init_weights(self, init_type = 'xavier'):
        if init_type == 'xavier':
            self.__xavier_initialization()
        else:
            self.__rand_initialization()

    def __rand_initialization(self):
        self.W1 = np.random.randn(self.dim_hidden1, self.dim_in).astype(np.float16)
        self.W2 = np.random.randn(self.dim_hidden2, self.dim_hidden1).astype(np.float16)
        self.W3 = np.random.randn(self.dim_out, self.dim_hidden2).astype(np.float16)
        self.W_skip = np.eye(self.dim_hidden2, self.dim_hidden1).astype(np.float16)

        self.b1 = np.random.randn(self.dim_hidden1).astype(np.float16)
        self.b2 = np.random.randn(self.dim_hidden2).astype(np.float16)
        self.b3 = np.random.randn(self.dim_out).astype(np.float16)

    def __xavier_initialization(self):
        std_in = 2.0 / (self.dim_in + self.dim_hidden1)
        self.W1 = np.random.uniform(-std_in, std_in, (self.dim_hidden1, self.dim_in)).astype(np.float16)
        self.b1 = np.random.uniform(-std_in, std_in, (self.dim_hidden1, 1)).astype(np.float16)

        std_hidden1 = 2.0 / (self.dim_hidden1 + self.dim_hidden2)
        self.W2 = np.random.uniform(-std_hidden1, std_hidden1, (self.dim_hidden2, self.dim_hidden1)).astype(np.float16)
        self.b2 = np.random.uniform(-std_hidden1, std_hidden1, (self.dim_hidden2, 1)).astype(np.float16)

        std_hidden2 = 2.0 / (self.dim_hidden2 + self.dim_out)
        self.W3 = np.random.uniform(-std_hidden2, std_hidden2, (self.dim_out, self.dim_hidden2)).astype(np.float16)
        self.b3 = np.random.uniform(-std_hidden2, std_hidden2, (self.dim_out, 1)).astype(np.float16)

        self.W_skip = np.eye(self.dim_hidden2, self.dim_in).astype(np.float16)

    def fit(self, X, y):
        self.X = X
        self.y = y
        _, self.m = X.shape
        self.__split_test_train_set()

    def predict(self, X_batch):
        if self.X is None:
            print('X is None. Run Fit with a valid input data.')
            raise RuntimeError('X is None. Run Fit with a valid input data.')

        return self._forward_pass(X_batch)

    def _forward_pass(self, X_batch):
        self.X1_hidden = self.W1.dot(X_batch) + self.b1
        self.X1_hidden_act = self.act_funcs[0](self.X1_hidden)

        self.X2_hidden = self.W2.dot(self.X1_hidden_act) + self.b2
        self.X_Skip = self.W_skip.dot(X_batch)
        self.X2_hidden_act = self.act_funcs[1](self.X2_hidden + self.X_Skip)

        self.X_out = self.W3.dot(self.X2_hidden_act)

        self.yp = self.act_funcs[-1](self.X_out)
        return self.yp

    def _bakward_pass_by_example(self, y_batch, y_result, batch):
        self.delta_out = (y_batch - y_result)
        self.delta_hidden_two = self.act_func_derivs[-1](self.X2_hidden) * np.dot(self.W3.T, self.delta_out)
        self.delta_hidden_one = self.act_func_derivs[-2](self.X1_hidden) * np.dot(self.W2.T, self.delta_hidden_two)

        self.delta_W_out = np.dot(self.delta_out, self.X2_hidden_act.T)
        self.delta_Skip = np.dot(self.delta_hidden_two, batch.T)
        self.delta_W_two = np.dot(self.delta_hidden_two, self.X1_hidden_act.T)
        self.delta_W_one = np.dot(self.delta_hidden_one, batch.T)

    def _weights_update(self, y_batch, y_result, batch):
        self._bakward_pass_by_example(y_batch, y_result, batch)
        self.W1 -= self.learning_rate * self.delta_W_one
        self.W2 -= self.learning_rate * self.delta_W_two
        self.W3 -= self.learning_rate * self.delta_W_out
        self.W_skip -= self.learning_rate * self.delta_Skip

        self.b3 -= self.learning_rate * np.sum(self.delta_out, axis=1, keepdims=True)
        self.b2 -= self.learning_rate * np.sum(self.delta_hidden_two, axis=1, keepdims=True)
        self.b1 -= self.learning_rate * np.sum(self.delta_hidden_one, axis=1, keepdims=True)

        self.delta_out = np.zeros(self.delta_out.shape)
        self.delta_hidden_two = np.zeros(self.delta_hidden_two.shape)
        self.delta_hidden_one = np.zeros(self.delta_hidden_one.shape)

        self.delta_W_out = np.zeros(self.delta_W_out.shape)
        self.delta_Skip = np.zeros(self.delta_Skip.shape)
        self.delta_W_two = np.zeros(self.delta_W_two.shape)
        self.delta_W_one = np.zeros(self.delta_W_one.shape)

    def train(self):
        self.epoch = 0
        self.train_err = []
        self.test_err = []
        while self.epoch < self.max_epoch:
            for batch_start in range(0, self.X_train.shape[1], self.batch_size):
                to_ind = min(batch_start + self.batch_size, self.X_train.shape[1])
                batch = self.X_train[:, batch_start: to_ind]
                self._forward_pass(batch)

                self._weights_update(self.yp, self.y_train[:, batch_start:to_ind], batch)

            predicts = np.zeros(self.predict(self.X_train).shape, dtype=np.int8)
            for i, el in enumerate(self.predict(self.X_train).transpose()):
                if el[0] > el[1]:
                    predicts[0, i] = 1
                else:
                    predicts[1, i] = 1
            self.train_err.append(error_function(self.y_train, predicts))
            self.test_err.append(error_function(self.y_test, self.predict(self.X_test)))
            self.epoch += 1

    def __split_test_train_set(self):
        self.X_train = self.X[:, :int(2 * self.m / 3)]
        self.y_train = self.y[:, :int(2 * self.m / 3)]
        self.X_test = self.X[:, int(2 * self.m / 3):]
        self.y_test = self.y[:, int(2 * self.m / 3):]


if __name__ == '__main__':
    y, X = data_prep('../data/wdbc.data')
    m, n = X.shape
    X = normalize(X, axis=0)
    #
    default_net = TheResNet(n, 10, 20, 2)
    default_net.fit(X.T, y)
    default_net.train()

    plt.figure(figsize=(12, 8))
    plt.xlabel('epoch')
    plt.plot( range(1, len(default_net.test_err) + 1, 1), default_net.test_err)
    plt.ylabel('accuracy')
    plt.title("Default Network Test accuracy")
    plt.show()

    tanh_out_net = TheResNet(n, 10, 20, 2, act_funcs=[tanh, relu, tanh])

    tanh_out_net.fit(X.T, y)
    tanh_out_net.train()

    plt.figure(figsize=(12, 8))
    plt.xlabel('epoch')
    plt.plot( range(1, len(tanh_out_net.test_err) + 1, 1), tanh_out_net.test_err)
    plt.ylabel('accuracy')
    plt.title("Tanh out Network Test accuracy")
    plt.show()


