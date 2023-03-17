import numpy as np


def data(path: str):
    data = np.genfromtxt(path)
    y, X = data[:, -1], data[:, :-1]
    return X, y


def n_bor(n: int, x, y, xi):
    dist = np.sum((x - xi)*(x - xi), axis=1)
    idx = np.argsort(dist)
    best_y = np.sum(y[idx[0:n]])
    return np.sign(best_y)


def KNN(n: int, x):
    yhat = np.zeros(len(x))
    for i in range(len(x)):
        yhat[i] = n_bor(n, x_train, y_train, x[i])
    return yhat


def error(yhat, y):
    return np.count_nonzero(yhat != y)/len(y)


if __name__ == "__main__":
    x_train, y_train = data("MOOC/techniques/HW4/hw4_nbor_train.dat")
    x_test, y_test = data("MOOC/techniques/HW4/hw4_nbor_test.dat")
    print(error(KNN(1, x_train), y_train))
    print(error(KNN(1, x_test), y_test))
    print(error(KNN(5, x_train), y_train))
    print(error(KNN(5, x_test), y_test))
