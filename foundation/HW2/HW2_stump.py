import numpy as np
SIZE = 20


def data():
    x = np.random.uniform(-1, 1, SIZE)
    noise = np.random.uniform(0, 1, SIZE)
    y = np.sign(x)
    y[noise < 0.2] *= -1
    return x, y


def error(t: float, x, y) -> int:
    res = 0
    fx = np.where(x > t, 1, -1)
    res = np.count_nonzero(fx != y)
    return res


def min_err():
    x, y = data()
    theta = x[0]
    min_err = error(theta, x, y)

    for temp_theta in x:
        temp_err = error(temp_theta, x, y)
        if temp_err < min_err:
            min_err = temp_err
            theta = temp_theta
    return theta, min_err


toterr = 0
errout = 0
n = 2000
for i in range(n):
    theta, err = min_err()
    toterr += err
    errout += error(theta, *data())
print(toterr/n/SIZE, errout/n/SIZE)
