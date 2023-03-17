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


def error(t: float, s: int, x, y) -> int:
    if s == 1:
        fx = np.where(x > t, 1, -1)
    elif s == -1:
        fx = np.where(x < t, 1, -1)
    else:
        raise ValueError
    return np.count_nonzero(fx != y)


def min_err(x, y):
    theta = x[0]
    min_err = error(theta, 1, x, y)

    for temp_theta in x:
        for temp_s in [1, -1]:
            temp_err = error(temp_theta, temp_s, x, y)
            if temp_err < min_err:
                min_err = temp_err
                theta = temp_theta
                s = temp_s
    return theta, min_err, s


x_train, y_train = data(
    "https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_math/hw2_train.dat")
x_test, y_test = data(
    "https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_math/hw2_test.dat")

best = list()
for i in range(len(x_train[0])):
    best.append(min_err(x_train[:, i], y_train))


theta, err, s, dim = *best[0], 0
for i in range(len(best)):
    if best[i][1] < err:
        theta = best[i][0]
        err = best[i][1]
        s = best[i][2]
        dim = i

print(theta, err/len(x_train), s, dim)
errout = error(theta, s, x_test[:, dim], y_test)
print(errout/len(x_test))
