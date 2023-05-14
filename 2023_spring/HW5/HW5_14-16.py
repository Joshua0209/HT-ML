import numpy as np
from libsvm.svmutil import *

np.random.seed(1126)


def relabel(digits: int, y):
    y = np.array(y)
    return (2 * (y == digits) - 1)


def valid_split(X, y, val_size):
    X, y = np.array(X), np.array(y)
    idx = np.random.permutation(len(X))
    x_train = X[idx[val_size:]]
    x_val = X[idx[:val_size]]
    y_train = y[idx[val_size:]]
    y_val = y[idx[:val_size]]
    return x_train, y_train, x_val, y_val


train_path = '2023_spring\HW5\letter.scale.tr'
test_path = '2023_spring\HW5\letter.scale.t'
y_train, x_train = svm_read_problem(train_path)
y_test, x_test = svm_read_problem(test_path)

y_train_7 = relabel(7, y_train)
y_test_7 = relabel(7, y_test)

Cs = [0.01, 0.1, 1, 10, 100]
Eout = np.zeros(len(Cs))
for i, C in enumerate(Cs):
    params = f'-s 0 -t 2 -c {C} -g 1 -q'
    m = svm_train(y_train_7, x_train, params)
    p_label, p_acc, p_val = svm_predict(y_test_7, x_test, m, '-q')
    Eout[i] = 100 - p_acc[0]

print(Eout)
print(Cs[np.argmin(Eout)])

gammas = [0.1, 1, 10, 100, 1000]
Eout = np.zeros(len(gammas))
for i, gamma in enumerate(gammas):
    params = f'-s 0 -t 2 -c 0.1 -g {gamma} -q'
    m = svm_train(y_train_7, x_train, params)
    p_label, p_acc, p_val = svm_predict(y_test_7, x_test, m, '-q')
    Eout[i] = 100 - p_acc[0]

print(Eout)
print(gammas[np.argmin(Eout)])

N = 1
best_gammas = np.zeros(len(gammas))
for j in range(N):
    x_train_, y_train_, x_val, y_val = valid_split(x_train, y_train_7, 200)
    E_val = np.zeros(len(gammas))
    for i, gamma in enumerate(gammas):
        params = f'-s 0 -t 2 -c 0.1 -g {gamma} -q'
        m = svm_train(y_train_, x_train_, params)
        p_label, p_acc, p_val = svm_predict(y_val, x_val, m, '-q')
        E_val[i] = 100 - p_acc[0]
    best_gammas[np.argmin(E_val)] += 1
    print(j, best_gammas)
print(best_gammas)
print(gammas[np.argmax(best_gammas)])


# [4.52 4.52 1.42 0.4  0.54]
# 10
# [4.52 4.52 4.02 4.52 4.52]
# 10
# [308.   0. 192.   0.   0.]
# 0.1
