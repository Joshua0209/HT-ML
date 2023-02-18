import numpy as np
import matplotlib.pyplot as plt


def data(path: str):
    data = np.genfromtxt(path)
    y, X = data[:, -1], data[:, :-1]
    return X, y


def g(x, s: int, dim: int, theta: float):
    return s*np.sign(x[:, int(dim)] - theta)


def error(x, y, w, s: int, dim: int, theta: float) -> float:
    return np.sum(np.inner((y != g(x, s, dim, theta)), w))


def stump():
    best_E = float("inf")
    for s in [-1, 1]:
        for dim in range(len(x_train[0])):
            thetas = np.sort(x_train[:, dim])
            thetas = [(thetas[i+1] + thetas[i]) /
                      2 for i in range(len(thetas)-1)]
            for theta in thetas:
                Ein = error(x_train, y_train, weights, s, dim, theta)
                if Ein < best_E:
                    best_E = Ein
                    best_s, best_dim, best_theta = s, dim, theta
    return best_s, best_dim, best_theta


def update_weights(t: int, s: int, dim: int, theta: float):
    ep = error(x_train, y_train, weights, s, dim, theta)/np.sum(weights)
    eps[t] = ep
    diamond = np.sqrt((1-ep)/ep)
    weights[y_train != g(x_train, s, dim, theta)] *= diamond
    weights[y_train == g(x_train, s, dim, theta)] /= diamond
    alphas[t] = np.log(diamond)


def final_G(x):
    res = np.zeros_like(x[:, 0])
    for t in range(T):
        res += alphas[t]*g(x, *G[t])
    return np.sign(res)


def final_E(x, y):
    return np.count_nonzero(y != final_G(x))/y.size


T = 300
x_train, y_train = data("hw2_adaboost_train.dat")
x_test, y_test = data("hw2_adaboost_test.dat")
weights = np.ones_like(y_train)/y_train.size
alphas = np.zeros(T)
eps = np.zeros(T)
G = np.zeros((T, 3))
for t in range(T):
    G[t] = stump()
    update_weights(t, *G[t])
    if t in [0, T-1]:
        print(t, final_E(x_train, y_train))  # Q12, 13
        print(t, np.sum(weights))  # Q14,15
        print(t, final_E(x_test, y_test))  # Q17,18
print(min(eps))  # Q16
