import numpy as np


def dataset(path: str) -> tuple[float]:
    data = np.genfromtxt(path)
    y, X = data[:, -1], data[:, :-1]
    return X, y


def error(t: float, s: int, x, y) -> float:
    fx = s*np.where(x > t, 1, -1)
    return np.count_nonzero(fx != y)/len(fx)


def min_err(x, y) -> tuple[float, float, int]:
    min_err = float("inf")
    tmp = np.sort(x)
    thetas = np.array([(tmp[i+1]+tmp[i])/2 for i in range(len(tmp)-1)]
                      + [float("-inf")])
    for theta in thetas:
        for s in [-1, 1]:
            err = error(theta, s, x, y)
            if err < min_err:
                min_err = err
                min_theta = theta
                min_s = s
    return min_theta, min_err, min_s


if __name__ == '__main__':
    x_train, y_train = dataset("2023_spring\HW2\hw2_train.dat")
    x_test, y_test = dataset("2023_spring\HW2\hw2_test.dat")
    best = []

    for i in range(len(x_train[0])):
        best.append(min_err(x_train[:, i], y_train))
    best = np.array(best)

    dim = np.argmin(best[:, 1])
    theta, s = best[dim, 0], best[dim, 2]
    Eout = error(theta, s, x_test[:, dim], y_test)
    ans17 = best[dim, 1]
    ans18 = Eout
    print(ans17, ans18)

    worst_dim = np.argmax(best[:, 1])
    theta, s = best[worst_dim, 0], best[worst_dim, 2]
    Eout = error(theta, s, x_test[:, worst_dim], y_test)
    ans19 = best[worst_dim, 1] - ans17
    ans20 = Eout - ans18
    print(ans19, ans20)

# 0.026041666666666668 0.078125
# 0.3020833333333333 0.34375
