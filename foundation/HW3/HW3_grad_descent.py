import numpy as np
import requests


def data(url: str):
    response = requests.get(url)
    data = response.text
    data = data.split('\n')
    x, y = list(), list()
    data.pop(-1)
    for i in data:
        x.append(i.split()[:-1])
        y.append(i.split()[-1])

    for i in range(len(x)):
        y[i] = int(y[i])
        for j in range(len(x[0])):
            x[i][j] = float(x[i][j])
    x, y = np.array(x), np.array(y)
    return x, y


def theta(s):
    return 1/(1+np.exp(-s))


def grad_Ein(w, x, y):
    return 1/len(y)*np.sum(theta(-y*np.matmul(w, x.T))*(-y*x.T), axis=1)


def error(w, x, y):
    yhat = np.matmul(w, x.T)
    return np.count_nonzero(np.sign(yhat) != np.sign(y))


x_train, y_train = data(
    "https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_algo/hw3_train.dat")
x_test, y_test = data(
    "https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_algo/hw3_test.dat")

ETA = 0.001
T = 2000
w = np.zeros_like(x_train[0])

for i in range(T):
    #     w = w - ETA*grad_Ein(w, x_train, y_train)
    x, y = x_train[i % len(x_train)], y_train[i % len(y_train)]
    w = w - ETA*theta(-y*np.matmul(w, x.T))*(-y*x.T)

Ein = error(w, x_train, y_train)
Eout = error(w, x_test, y_test)
print(Ein/len(x_train), Eout/len(x_test))
