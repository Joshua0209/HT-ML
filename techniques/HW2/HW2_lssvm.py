import numpy as np
import matplotlib.pyplot as plt


def data(path: str):
    data = np.genfromtxt(path)
    y, X = data[:, -1], data[:, :-1]
    return X, y


def kernel(gamma, x1, x2):
    return np.exp(-gamma*np.linalg.norm(x1-x2)**2)


def K_matrix(gamma, x1, x2):
    N = len(x1)
    M = len(x2)
    K = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            K[i, j] = kernel(gamma, x1[i], x2[j])
    return K


def beta(l, K, y):
    return np.linalg.inv(np.eye(len(K))*l+K)@y


def error(y, K, beta):
    return np.count_nonzero(y != np.sign(beta@K))/y.size


N = 400
x, y = data("hw2_lssvm_all.dat")
x_train, y_train = x[:N], y[:N]
x_test, y_test = x[N:], y[N:]
gammas = np.array([32, 2, 0.125])
lambdas = np.array([0.001, 1, 1000])
best_Ein = float("inf")
best_Eout = float("inf")
for gamma in gammas:
    K_train = K_matrix(gamma, x_train, x_train)
    K_test = K_matrix(gamma, x_train, x_test)
    for l in lambdas:
        b = beta(l, K_train, y_train)
        Ein = error(y_train, K_train, b)
        best_Ein = min(best_Ein, Ein)
        Eout = error(y_test, K_test, b)
        best_Eout = min(best_Eout, Eout)
print(best_Ein, best_Eout)
