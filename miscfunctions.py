# Chase Brown
# SID 106015389
# DeepLearning PA 3: CNN

import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from scipy.special import expit
import time


def reshape_x(x_array):
    new_x = []
    for row in x_array:
        img = row.reshape(28, 28)
        new_x.append(img)
    x = np.array(new_x)
    return x


def soft_max(x):
    """
    Stable softmax function as to not over run the float max of python 10^308
    :param x:
    :return:
    """
    x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return x/np.sum(x, axis=1, keepdims=True)


def crossEntropyLoss(x, y):
    # X is the output from fully connected layer (num_examples x num_classes)
    # y is labels (num_examples x 1)
    m = y.shape[0]
    p = soft_max(x)
    log_likelihood = -np.log(p[range(m), y])
    loss = np.sum(log_likelihood) / m

    # and its derivative
    dx = p.copy()
    dx[range(m), y] = dx[range(m), y] - 1
    dx = dx / m

    return loss, dx


def get_minibatches(x, y, minibatch_size, shuf=True):
    m = x.shape[0]
    minibatches = []
    if shuf:
        x, y = shuffle(x, y)
    for i in range(0, m, minibatch_size):
        x_batch = x[i:i + minibatch_size, :, :, :]
        y_batch = y[i:i + minibatch_size, ]
        minibatches.append((x_batch, y_batch))
    return minibatches


def accuracy(y_true, y_pred):
    return np.mean(y_pred == y_true)  # both are not one hot encoded


def vanilla_update(params, grads, learning_rate=0.01):
    for param, grad in zip(params, reversed(grads)):
        for i in range(len(grad)):
            param[i] += - learning_rate * grad[i]


# Stochastic gradient descent
def sgd(nnet, x_train, y_train, minibatch_size, epoch, learning_rate, verbose=True, x_test=None, y_test=None):
    minibatches = get_minibatches(x_train, y_train, minibatch_size)
    for i in range(epoch):
        t1 = time.time()
        loss = 0
        if verbose:
            print("Epoch {0}".format(i + 1))
        for x_mini, y_mini in minibatches:
            loss, grads = nnet.train_step(x_mini, y_mini)
            vanilla_update(nnet.params, grads, learning_rate=learning_rate)
        if verbose:
            train_acc = accuracy(y_train, nnet.predict(x_train))
            test_acc = accuracy(y_test, nnet.predict(x_test))
            t2 = time.time()
            print("Loss = {0} | Training Accuracy = {1} | Test Accuracy = {2}".format(loss, train_acc, test_acc))
            print("Epoch {0} took {1}".format(i, t2-t1))
    return nnet


class TanH:
    def __init__(self):
        self.params = []

    def forward(self, x):
        """
        Passes the Matrix X into the tanh function for activation.
        :param x: nxm Matrix
        :return:  nxm matrix
        """
        out = np.tanh(x)
        self.out = out
        return out

    def backward(self, dout):
        delta_x = dout * (1 - self.out ** 2)
        return delta_x, []


class ReLU:
    def __init__(self):
        self.params = []

    def forward(self, x):
        self.x = x
        return np.maximum(0, x)

    def backward(self, dout):
        dx = dout.copy()
        dx[self.x <= 0] = 0
        return dx, []

class sigmoid():
    def __init__(self):
        self.params = []

    def forward(self, x):
        out = expit(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)
        return dx, []


def momentum_update(velocity, params, grads, learning_rate=0.01, mu=0.9):
    for v, param, grad in zip(velocity, params, reversed(grads)):
        for i in range(len(grad)):
            v[i] = mu*v[i] + learning_rate * grad[i]
            param[i] -= v[i]


def sgd_momentum(nnet, x_train, y_train, minibatch_size, epoch, lr, mu, x_test, y_test, verbose=True):
    minibatches = get_minibatches(x_train, y_train, minibatch_size)
    for i in range(epoch):
        loss = 0
        velocity = []
        for param_layer in nnet.params:
            p = [np.zeros_like(param) for param in list(param_layer)]
            velocity.append(p)

        if verbose:
            print('Epoch {0}'.format(i + 1))
        for x_mini, y_mini in minibatches:
            loss, grads = nnet.train_step(x_mini, y_mini)
            momentum_update(velocity, nnet.params, grads, learning_rate=lr, mu=mu)
        if verbose:
            train_acc = accuracy_score(y_train, nnet.predict(x_train))
            test_acc = accuracy_score(y_test, nnet.predict(x_test))
            print('Loss = {0} | Training Accuracy = {1} | Test Accuracy = {2}'.format(loss, train_acc, test_acc))

    return nnet