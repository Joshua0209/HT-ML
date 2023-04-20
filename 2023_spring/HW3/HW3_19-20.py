import numpy as np


def data(path):
    data = np.genfromtxt(path)
    y, X_data = data[:, -1], data[:, :-1]
    return X_data, y


def transform(X, Q):
    X_output = np.ones_like(X[:, 0])[np.newaxis, :]
    for i in range(1, Q+1):
        X_output = np.vstack((X_output, (X**i).T))
    return X_output.T


def error(w, X, y):
    return np.count_nonzero(y != np.sign(X@w))/len(X)


if __name__ == "__main__":
    x_train, y_train = data("2023_spring\HW3\hw3_train.dat")
    x_test, y_test = data("2023_spring\HW3\hw3_test.dat")

    # Q19: Quadratic regression Q=2
    x_train_2 = transform(x_train, 2)
    x_test_2 = transform(x_test, 2)
    w19 = np.linalg.inv((x_train_2.T)@x_train_2)@(x_train_2.T)@y_train
    print(abs(error(w19, x_train_2, y_train) - error(w19, x_test_2, y_test)))

    # Q20: Quadratic regression Q=8
    x_train_8 = transform(x_train, 8)
    x_test_8 = transform(x_test, 8)
    w19 = np.linalg.inv((x_train_8.T)@x_train_8)@(x_train_8.T)@y_train
    print(abs(error(w19, x_train_8, y_train) - error(w19, x_test_8, y_test)))

# 0.08249999999999999
# 0.415
