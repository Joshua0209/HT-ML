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


def valid_split(X, y, val_size):
    idx = np.random.permutation(len(X))
    x_train = X[idx[val_size:]]
    x_val = X[idx[:val_size]]
    y_train = y[idx[val_size:]]
    y_val = y[idx[:val_size]]
    return x_train, y_train, x_val, y_val


def error_val(x_train, y_train, x_val, y_val):
    err = np.full(5, np.inf)
    for i in range(len(log_lambdas)):
        prob = problem(y_train, x_train)
        params = parameter(f"-s 0 -c {1/10**log_lambdas[i]/2} -e 0.000001 -q")
        m = train(prob, params)
        p_label, p_acc, p_val = predict(y_val, x_val, m, "-q")
        err[i] = 100 - p_acc[0]
        if err[i] == np.min(err):
            best_m = m

    return err, best_m


if __name__ == "__main__":
    Q = 4
    x_data, y_data = data("2023_spring\HW4\hw4_train.dat", Q)
    x_test, y_test = data("2023_spring\HW4\hw4_test.dat", Q)
    N, train_size, val_size = 256, 120, 80
    log_lambdas = [6, 3, 0, -3, -6]

    ans14 = np.zeros(5)
    ans15 = 0
    ans16 = 0
    for i in range(N):
        # Q14: select best lambda by validation
        x_train, y_train, x_val, y_val = valid_split(x_data, y_data, val_size)
        tmp_err, tmp_m = error_val(x_train, y_train, x_val, y_val)
        ans14[np.argmin(tmp_err)] += 1

        # Q15: estimate E_out with w_lambda*^- (run on D_train)
        p_label, p_acc, p_val = predict(y_test, x_test, tmp_m, "-q")
        ans15 += 100 - p_acc[0]

        # Q16: estimate E_out with w_lambda* (run on D_train \cup D_val)
        tmp_lambda = log_lambdas[np.argmin(tmp_err)]
        # print(tmp_lambda)
        prob = problem(y_data, x_data)
        params = parameter(f"-s 0 -c {1/10**tmp_lambda/2} -e 0.000001 -q")
        m = train(prob, params)
        p_label, p_acc, p_val = predict(y_test, x_test, m, "-q")
        # print(p_acc)
        ans16 += 100 - p_acc[0]

    print(log_lambdas[np.argmax(ans14)], np.max(ans14))
    print(ans15/N)
    print(ans16/N)

# Q = 4
# 3 162.0
# 17.610937500000002
# 14.869531249999984


# Q = 2
# 0 197.0
# 19.376562499999995
# 17.640625
