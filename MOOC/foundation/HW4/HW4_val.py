import numpy as np
import requests


def data(url: str):
    response = requests.get(url)
    data = response.text
    data = data.split('\n')
    x, y = list(), list()
    data.pop(-1)
    for i in data:
        x.append([1]+i.split()[:-1])
        y.append(i.split()[-1])

    for i in range(len(x)):
        y[i] = int(y[i])
        for j in range(len(x[0])):
            x[i][j] = float(x[i][j])
    x, y = np.array(x), np.array(y)
    return x, y


def ridge_regression(x, y, Lambda):
    hat = np.matmul(np.linalg.inv(
        np.matmul(x, x.T) + Lambda*np.eye(len(x))), x)
    w = np.inner(hat.T, y)
    return w


def error(x, y, w):
    return np.count_nonzero(y != np.sign(np.inner(w, x)))


def best_lambda(x, y):
    LAMBDA = np.linspace(2, -10, 13)
    best_E = error(x_train, y_train, ridge_regression(
        x_train, y_train, 10**LAMBDA[0]))
    best_l = LAMBDA[0]
    for i in range(len(LAMBDA)):
        w = ridge_regression(x_train, y_train, 10**LAMBDA[i])
        E = error(x, y, w)
        if E < best_E:
            best_E = E
            best_l = LAMBDA[i]
    return best_l


x_train_all, y_train_all = data(
    "https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_algo/hw4_train.dat")
x_test, y_test = data(
    "https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_algo/hw4_test.dat")


x_val, y_val = x_train_all[120:], y_train_all[120:]
x_train, y_train = x_train_all[:120], y_train_all[:120]

best_l = best_lambda(x_val, y_val)
w = ridge_regression(x_train_all, y_train_all, 10**best_l)

Ein = error(x_train, y_train, w)
# Eval = error(x_val, y_val, w)
Eout = error(x_test, y_test, w)
print(best_l, Ein/len(x_train), Eout/len(x_test))
