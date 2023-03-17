import requests
import numpy as np

response = requests.get(
    'https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_math/hw1_18_train.dat')
data = response.text
data = data.split('\n')
x_train = list()
data.pop(-1)
for i in data:
    x_train.append([1]+i.split())

for i in range(len(x_train)):
    for j in range(len(x_train[0])):
        x_train[i][j] = float(x_train[i][j])

response = requests.get(
    'https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_math/hw1_18_test.dat')
data = response.text
data = data.split('\n')
x_test = list()
data.pop(-1)
for i in data:
    x_test.append([1]+i.split())

for i in range(len(x_test)):
    for j in range(len(x_test[0])):
        x_test[i][j] = float(x_test[i][j])

x_test = np.array(x_test)
x_train = np.array(x_train)

# train


def error(w, data):
    res = 0
    for x in data:
        if np.sign(np.dot(w, x[:-1])) != np.sign(x[-1]):
            res += 1
    return res


def w_pocket():
    w = np.array([0., 0., 0., 0., 0.])
    w_hat = w
    min_err = error(w, x_train)
    halt = 0
    while halt < 50:
        np.random.shuffle(x_train)
        for x in x_train:
            if np.sign(np.dot(w, x[:-1])) != np.sign(x[-1]):
                w = w + x[:-1]*x[-1]
                err = error(w, x_train)
                if err < min_err:
                    min_err = err
                    w_hat = w
                halt += 1
                break

    return w_hat


# test


n = 200
err = 0
for j in range(n):
    err += error(w_pocket(), x_test)
    print(j)
print(err, err/n/len(x_test))
