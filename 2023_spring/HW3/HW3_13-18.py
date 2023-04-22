import numpy as np


def data(path):
    data = np.genfromtxt(path)
    y, X_data = data[:, -1], data[:, :-1]
    tmp = np.ones(len(X_data))
    X = np.vstack((tmp, X_data.T)).T
    return X, y


def theta(s):
    return 1/(1+np.exp(-s))


def error(w, X, y, type):
    if type == "0/1":
        return np.count_nonzero(y != np.sign(X@w))/len(X)
    elif type == "sqr":
        return np.sum((X@w - y)**2)/len(X)
    elif type == "ce":
        return np.sum(np.log(1 + np.exp(-y*(X@w))))/len(X)


def grad(w, X, y, type):
    if type == "sqr":
        return 2*(np.outer(X, X)@w - (X.T)*y)
    elif type == "ce":
        return theta(-y*(X@w))*(-y*X)


def SGD(type, w0):
    toterr = 0.
    for i in range(N):
        w = w0.copy()
        for t in range(T):
            idx = np.random.randint(0, len(x_train))
            w += -ETA*grad(w, x_train[idx], y_train[idx], type)
        toterr += error(w, x_train, y_train, type)
        del w
    return toterr/N


def Eout_estimate(type, w0):
    output = 0.
    for i in range(N):
        w = w0.copy()
        for t in range(T):
            idx = np.random.randint(0, len(x_train))
            w += -ETA*grad(w, x_train[idx], y_train[idx], type)
        output += abs(error(w, x_train, y_train, "0/1") -
                      error(w, x_test, y_test, "0/1"))
        del w
    return output/N


if __name__ == "__main__":
    x_train, y_train = data("2023_spring\HW3\hw3_train.dat")
    x_test, y_test = data("2023_spring\HW3\hw3_test.dat")
    # Q13: linear regression
    w13 = np.linalg.inv((x_train.T)@x_train)@(x_train.T)@y_train
    print(error(w13, x_train, y_train, "sqr"))

    # Q14: SGD linear regression
    ETA, T, N = 0.001, 800, 1000
    print(SGD("sqr", np.zeros(len(x_train[0]))))

    # Q15: SGD logistic regression
    print(SGD("ce", np.zeros(len(x_train[0]))))

    # Q16: SGD logistic regression with w0 = w_lin = w_13
    print(SGD("ce", w13))

    # Q17
    print(Eout_estimate("ce", w13))

    # Q18
    print(abs(error(w13, x_train, y_train, "0/1") -
              error(w13, x_test, y_test, "0/1")))

# 0.7922347761105571
# 0.8234241910443619
# 0.6573118958314371
# 0.6052453923482037
# 0.030770000000000016
# 0.040000000000000036
