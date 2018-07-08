import numpy as np
import pandas as pd

from ResNetLike.src.functions import tanh, relu, softmax, relu_deriv, tanh_deriv, error_function
from sklearn.preprocessing import normalize


class TheResNet(object):
    def __init__(self, dim_in, dim_hidden1, dim_hidden2, dim_out, learning_rate=7*10**(-3), batch_size=10,
                 rand_seed=42, max_epochs=200):
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
        self.X1_hidden_act = tanh(self.X1_hidden)

        self.X2_hidden = self.W2.dot(self.X1_hidden_act) + self.b2
        self.X_Skip = self.W_skip.dot(X_batch)
        self.X2_hidden_act = relu(self.X2_hidden + self.X_Skip)

        self.X_out = self.W3.dot(self.X2_hidden_act)

        self.yp = softmax(self.X_out)
        return self.yp

    def _bakward_pass_by_example(self, y_batch, y_result, batch):
        self.delta_out = (y_batch - y_result)
        self.delta_hidden_two = relu_deriv(self.X2_hidden) * np.dot(self.W3.T, self.delta_out)
        self.delta_hidden_one = tanh_deriv(self.X1_hidden) * np.dot(self.W2.T, self.delta_hidden_two)

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
            #print('the epoch is {}'.format(self.epoch))
            for batch_start in range(0, self.X_train.shape[1], self.batch_size):
                to_ind = min(batch_start + self.batch_size, self.X_train.shape[1])
                batch = self.X_train[:, batch_start: to_ind]
                #print(batch.shape)
                self._forward_pass(batch)
                #err = error_function(self.y_train[:, batch_start: to_ind], self.yp)
                self._weights_update(self.yp, self.y_train[:, batch_start:to_ind], batch)

            predicts = np.zeros(self.predict(self.X_train).shape, dtype=np.int8)
            for i, el in enumerate(self.predict(self.X_train).transpose()):
                if el[0] > el[1]:
                    predicts[0, i] = 1
                else:
                    predicts[1, i] = 1
            self.train_err.append(error_function(self.y_train, predicts))
            print(self.train_err[-1])
            #self.test_err.append(error_function(self.y_test, self.predict(self.X_test)))
            self.epoch += 1


    def __split_test_train_set(self):
        self.X_train = self.X[:, :int(2 * self.m / 3)]
        self.y_train = self.y[:, :int(2 * self.m / 3)]
        self.X_test = self.X[:, int(2 * self.m / 3):]
        self.y_test = self.y[:, int(2 * self.m / 3):]


def data_prep(path_csv):
    df = pd.read_csv(path_csv, header=None)
    y_real = np.zeros((2, df.shape[0]))
    for i in range(df.shape[0]):
        idx = 0
        if df[1].values[i] == 'M':
            idx = 1
        y_real[idx, i] = 1
    return (y_real, df[np.arange(2, df.shape[-1], 1)].values)


if __name__ == '__main__':
    y, X = data_prep('../data/wdbc.data')
    m, n = X.shape
    net = TheResNet(n, 10, 20, 2)
    X = normalize(X, axis=0)
    net.fit(X.T, y)
    net.train()


