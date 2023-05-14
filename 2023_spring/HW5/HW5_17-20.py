import numpy as np
import matplotlib.pyplot as plt
from libsvm.svmutil import *


def data(path: str):
    y_data, X_data = svm_read_problem(path, return_scipy=True)
    X_data = X_data.toarray()
    idx11 = y_data == 11
    idx26 = y_data == 26
    y = y_data[idx11 | idx26]
    y[y == 11] = 1
    y[y == 26] = -1
    X = X_data[idx11 | idx26]
    return X, y


def g(x, s: int, dim: int, theta: float):
    yhat = s*np.sign(x[:, int(dim)] - theta)
    yhat[yhat == 0] = 1
    return yhat


def error_w(x, y, s: int, dim: int, theta: float, w) -> float:
    return np.sum(np.inner((y != g(x, s, dim, theta)), w))


class AdaBoost():
    def __init__(self, train_path, test_path, T):
        self.x_train, self.y_train = data(train_path)
        self.x_test, self.y_test = data(test_path)
        self.weights = np.ones_like(self.y_train)/self.y_train.size
        self.alphas = np.zeros(T)
        self.eps = np.zeros(T)
        self.G = np.zeros((T, 3))
        self.Ein = np.zeros(T)
        self.T = T

    def stump(self, t: int):
        best_E = np.inf
        for s in [-1, 1]:
            for dim in range(len(self.x_train[0])):
                thetas = np.sort(self.x_train[:, dim])
                thetas = [(thetas[i+1] + thetas[i]) /
                          2 for i in range(len(thetas)-1)] + [-np.inf]
                for theta in thetas:
                    Ein = error_w(self.x_train, self.y_train,
                                  s, dim, theta, self.weights)
                    if Ein < best_E:
                        best_E = Ein
                        best_s, best_dim, best_theta = s, dim, theta
        self.G[t] = np.array([best_s, best_dim, best_theta])

    def update_weights(self, t: int):
        ep = error_w(self.x_train, self.y_train, *
                     self.G[t], self.weights)/np.sum(self.weights)
        self.eps[t] = ep
        diamond = np.sqrt((1-ep)/ep)
        self.weights[self.y_train != g(self.x_train, *self.G[t])] *= diamond
        self.weights[self.y_train == g(self.x_train, *self.G[t])] /= diamond
        self.alphas[t] = np.log(diamond)

    def final_G(self, x, T=1000):
        res = np.zeros_like(x[:, 0])
        for t in range(T):
            res += self.alphas[t]*g(x, *self.G[t])
        return np.sign(res)

    def train(self):
        for t in range(self.T):
            self.stump(t)
            self.update_weights(t)
            self.Ein[t] = np.sum(self.y_train != g(
                self.x_train, *self.G[t]))/self.y_train.size
            print(t)

    def final_E(self, y, x, t: int = 1000):
        return np.sum(y != self.final_G(x, t))/y.size


if __name__ == '__main__':
    T = 1000
    train_path = "2023_spring\HW5\letter.scale.tr"
    test_path = "2023_spring\HW5\letter.scale.t"
    adb = AdaBoost(train_path, test_path, T)
    adb.train()
    print(min(adb.Ein))
    print(max(adb.Ein))
    print(adb.final_E(adb.y_train, adb.x_train))
    print(adb.final_E(adb.y_test, adb.x_test))

# 0.09846547314578005
# 0.571611253196931
# 0.0
# 0.002793296089385475
