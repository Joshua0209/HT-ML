import numpy as np


def init(x, k: int):
    idx = np.random.randint(0, len(x), k)
    return x[idx]


def opt_S(x, mu):  # category
    tmp = np.zeros((len(mu), len(x)))
    for i in range(len(mu)):
        tmp[i] = np.sum((x - mu[i])*(x - mu[i]), axis=1)
    return np.argmin(tmp, axis=0)


def opt_mu(x, S):  # center
    mu = np.zeros((len(np.unique(S)), len(x[0])))
    for i in range(len(np.unique(S))):
        s = np.unique(S)[i]
        mu[i] = np.sum(x[S == s], axis=0)/len(x[S == s])
    return np.array(mu)


def error(x, S, mu):
    err = 0
    for i in range(len(mu)):
        s = np.unique(S)[i]
        err += np.sum((x[S == s] - mu[i])**2)
    return err/len(x)


def kMeans(x, k: int):
    mu = init(x, k)
    S = np.random.random(len(x))
    pre_s = np.ones(len(x))

    while (pre_s != S).any():
        pre_s = S
        S = opt_S(x, mu)
        mu = opt_mu(x, S)
    return error(x, S, mu)


if __name__ == '__main__':
    x = np.genfromtxt("hw4_nolabel_train.dat")
    N = 500
    ks = [2, 10]
    for k in ks:
        err = 0
        for i in range(N):
            err += kMeans(x, k)
        print(k, err/N)
