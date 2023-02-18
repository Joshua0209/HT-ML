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


def error_cv(x, y, Lambda, V_fold):
    err = 0
    for v in range(V_fold):
        x_train = np.concatenate(
            (x[:len(x) // V_fold*v], x[len(x) // V_fold*(v+1):]))
        y_train = np.concatenate(
            (y[:len(y) // V_fold*v], y[len(y) // V_fold*(v+1):]))
        x_val = x[len(x) // V_fold * v: len(x) // V_fold*(v+1)]
        y_val = y[len(y) // V_fold * v: len(y) // V_fold*(v+1)]
        w = ridge_regression(x_train, y_train, 10**Lambda)
        err += error(x_val, y_val, w)
    return err


def best_lambda(x, y, V_fold):
    LAMBDA = np.linspace(2, -10, 13)
    best_E = len(x)
    best_l = LAMBDA[0]
    for i in range(len(LAMBDA)):
        E = error_cv(x, y, LAMBDA[i], V_fold)
        if E < best_E:
            best_E = E
            best_l = LAMBDA[i]
    return best_l


x_train_all, y_train_all = data(
    "https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_algo/hw4_train.dat")
x_test, y_test = data(
    "https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_algo/hw4_test.dat")

V_fold = 5
best_l = best_lambda(x_train_all, y_train_all, V_fold)
# Ecv = error_cv(x_train_all, y_train_all, best_l, V_fold)
# print(best_l, Ecv/(len(x_train_all)*(V_fold-1)/V_fold))
w = ridge_regression(x_train_all, y_train_all, 10**best_l)
Ein = error(x_train_all, y_train_all, w)
Eout = error(x_test, y_test, w)
print(Ein/len(x_train_all), Eout/len(x_test))
