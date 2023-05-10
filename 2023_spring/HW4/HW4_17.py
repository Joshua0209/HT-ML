import itertools
import numpy as np
from liblinear.liblinearutil import *

np.random.seed(1126)


def transform(X, Q):
    X_output = np.ones_like(X[:, 0])[np.newaxis, :]
    for L in range(1, Q+1):
        for subset in itertools.combinations_with_replacement(range(len(X[0])), L):
            tmp = np.ones_like(X[:, 0])[np.newaxis, :]
            for idx in subset:
                tmp = tmp*X[:, idx]
            X_output = np.vstack((X_output, tmp))
    return X_output.T


def data(path, Q):
    data = np.genfromtxt(path)
    y, X_data = data[:, -1], transform(data[:, :-1], Q)
    return X_data, y


def error_cv(x, y, V_fold, i):
    err = 0
    for v in range(V_fold):
        x_train = np.concatenate(
            (x[:len(x) // V_fold*v], x[len(x) // V_fold*(v+1):]))
        y_train = np.concatenate(
            (y[:len(y) // V_fold*v], y[len(y) // V_fold*(v+1):]))
        x_val = x[len(x) // V_fold * v: len(x) // V_fold*(v+1)]
        y_val = y[len(y) // V_fold * v: len(y) // V_fold*(v+1)]
        prob = problem(y_train, x_train)
        params = parameter(f"-s 0 -c {1/10**log_lambdas[i]/2} -e 0.000001 -q")
        m = train(prob, params)
        p_label, p_acc, p_val = predict(y_val, x_val, m, "-q")
        err += 100-p_acc[0]
    return err/V_fold


if __name__ == "__main__":
    Q = 4
    x_train, y_train = data("2023_spring\HW4\hw4_train.dat", Q)
    x_test, y_test = data("2023_spring\HW4\hw4_test.dat", Q)
    N, train_size, val_size = 256, 120, 80
    V_fold = 5
    log_lambdas = [6, 3, 0, -3, -6]

    # Q17
    err = np.zeros(5)

    for j in range(N):
        for i in range(len(log_lambdas)):
            idx = np.random.permutation(len(x_train))
            err[i] += error_cv(x_train[idx], y_train[idx], V_fold, i)
        print(err/(j+1))

    print(err/N)
    print(np.min(err)/N)

# [25.90039062 13.33789062 14.70703125 16.41992188 19.6875]
# 13.337890625
