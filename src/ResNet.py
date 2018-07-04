import numpy as np

from functions import tanh, relu, softmax


class TheResNet(object):
    def __init__(self, dim_in, dim_hidden1, dim_hidden2, dim_out, learning_rate = 0.05, batch_size = 10,
                 rand_seed = 42):
        self.dim_in = dim_in
        self.dim_hidden1 = dim_hidden1
        self.dim_hidden2 = dim_hidden2
        self.dim_out = dim_out
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.rand_seed = rand_seed
        self._init_weights(init_type='xavier')
        self.X = None


    def _init_weights(self, init_type = 'xavier'):
        if init_type == 'xavier':
            self.__xavier_initialization()
        else:
            self.__rand_initialization()

    def __rand_initialization(self):
        self.W1 = np.random.randn(self.dim_hidden1, self.dim_in).astype(np.float32)
        self.W2 = np.random.randn(self.dim_hidden2, self.dim_hidden1).astype(np.float32)
        self.W3 = np.random.randn(self.dim_out, self.dim_hidden2).astype(np.float32)
        self.W_skip = np.eye(self.dim_hidden2, self.dim_hidden1).astype(np.float32)

        self.b1 = np.random.randn(self.dim_hidden1).astype(np.float32)
        self.b2 = np.random.randn(self.dim_hidden2).astype(np.float32)
        self.b3 = np.random.randn(self.dim_out).astype(np.float32)

    def __xavier_initialization(self):
        std_in = 2.0 / (self.dim_in + self.dim_hidden1)
        self.W1 = np.random.uniform(-std_in, std_in, (self.dim_hidden1, self.dim_in)).astype(np.float32)
        self.b1 = np.random.uniform(-std_in, std_in, (self.dim_hidden1, 1)).astype(np.float32)

        std_hidden1 = 2.0 / (self.dim_hidden1 + self.dim_hidden2)
        self.W2 = np.random.uniform(-std_hidden1, std_hidden1, (self.dim_hidden2, self.dim_hidden1)).astype(np.float32)
        self.b2 = np.random.uniform(-std_hidden1, std_hidden1, (self.dim_hidden2, 1)).astype(np.float32)

        std_hidden2 = 2.0 / (self.dim_hidden2 + self.dim_out)
        self.W3 = np.random.uniform(-std_hidden2, std_hidden2, (self.dim_out, self.dim_hidden2)).astype(np.float32)
        self.b3 = np.random.uniform(-std_hidden2, std_hidden2, (self.dim_out, 1)).astype(np.float32)

        self.W_skip = np.eye(self.dim_hidden2, self.dim_in).astype(np.float32)

    # def fit(self, X, n_iter = 1000):
    #  logger = {}
    #  logger['iteration'] = []
    #  logger['loss_iteration'] = []
    #
    #  for t in range(n_iter):
    #
    #      # forward pass
    #      x_hidden1 = X.mm(self.W1) + b1
    #      x_hidden_act1 = torch.tanh(x_hidden1)
    #
    #      x_hidden2 = x_hidden_act1.mm(w2) + b2
    #      x_hidden_act2 = torch.relu(x_hidden2)
    #
    #      y_out = x_hidden_act2.mm(w3) + b3
    #      y_pred = F.softmax(y_out)  # YOUR CODE HERE
    #
    #      # compute loss
    #      loss = criterion(y_pred, y)
    #
    #      # backprop
    #      loss.backward()
    #
    #      # update weights using gradient descent
    #      w1.data -= learning_rate * w1.grad
    #      w2.data -= learning_rate * w2.grad
    #      b1.data -= learning_rate * b1.grad
    #      b2.data -= learning_rate * b2.grad
    #
    #      # manually zero the gradients
    #      w1.grad.zero_()
    #      w2.grad.zero_()
    #      b1.grad.zero_()
    #      b2.grad.zero_()
    #
    #      # reporting & logging
    #      if t % 100 == 0:
    #          print(t, loss.item())
    #
    #      logger['iteration'] += [t]
    #      logger['loss_iteration'] += [loss.item()]

    def fit(self, X):
        self.X = X

    def predict(self, X_batch):
        if self.X is None:
            print('X is None. Run Fit with a valid input data.')
            raise RuntimeError('X is None. Run Fit with a valid input data.')

        return np.argmax(self._forward_pass(X_batch))

    def _forward_pass(self, X_batch):
        self.X1_hidden = self.W1.dot(X_batch) + self.b1
        self.X1_hidden_act = tanh(self.X1_hidden)

        self.X2_hidden = self.W2.dot(self.X1_hidden_act) + self.b2
        self.X_Skip = self.W_skip.dot(X_batch)
        self.X2_hidden_act = relu(self.X2_hidden + self.X_Skip)

        self.X_out = self.W3.dot(self.X2_hidden_act)

        self.yp = softmax(self.X_out)

        return self.yp


if __name__ == '__main__':
    net = TheResNet(3, 100, 100, 2)
    rand_d = np.random.randn(3, 20).astype(np.float32)
    net.fit(rand_d)
    net.predict(rand_d)