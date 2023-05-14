import numpy as np
from libsvm.svmutil import *


def relabel(digits: int, y):
    y = np.array(y)
    return (2 * (y == digits) - 1)


train_path = '2023_spring\HW5\letter.scale.tr'
test_path = '2023_spring\HW5\letter.scale.t'

y_train, x_train = svm_read_problem(train_path)
y_test, x_test = svm_read_problem(test_path)
m = svm_train(relabel(1, y_train), x_train, '-s 0 -t 0 -c 1')
print(x_train[0])

nr_feats = len(x_train[0])
w = np.zeros(nr_feats)
SVs = m.get_SV()
SV_coefs = m.get_sv_coef()
for i in range(m.get_nr_sv()):
    sv = np.array([SVs[i][j+1] for j in range(nr_feats)])
    w += SV_coefs[i][0] * sv

print(np.linalg.norm(w))


digits = [2, 3, 4, 5, 6]
Ein = np.zeros(len(digits))
nr_SVs = np.zeros(len(digits))
for i, digit in enumerate(digits):
    params = '-s 0 -t 1 -d 2 -c 1 -g 1 -r 1 -q'
    m = svm_train(relabel(digit, y_train), x_train, params)
    p_label, p_acc, p_val = svm_predict(
        relabel(digit, y_train), x_train, m, '-q')
    Ein[i] = 100 - p_acc[0]
    nr_SVs[i] = m.get_nr_sv()

print(Ein)
print(digits[np.argmax(Ein)])
print(nr_SVs)
print(min(nr_SVs))


# 6.309673609961579

# [1.13333333 0.67619048 0.96190476 1.48571429 1.12380952]
# 5

# [588. 368. 499. 642. 503.]
# 368.0
