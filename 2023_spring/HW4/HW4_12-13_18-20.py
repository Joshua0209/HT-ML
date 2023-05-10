import itertools
import numpy as np
from liblinear.liblinearutil import *


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


if __name__ == "__main__":
    Q = 4
    x_train, y_train = data("2023_spring\HW4\hw4_train.dat", Q)
    x_test, y_test = data("2023_spring\HW4\hw4_test.dat", Q)

    log_lambdas = np.array([6, 3, 0, -3, -6], dtype=float)
    ans = np.zeros((3, 5))  # [[Q12], [Q13], [Q18]] C = 1/lambda/2
    for i in range(len(log_lambdas)):
        prob = problem(y_train, x_train)

        params = parameter(f"-s 0 -c {1/10**log_lambdas[i]/2} -e 0.000001 -q")
        m = train(prob, params)
        # Q12: L2 estimate with Eout
        p_label, p_acc, p_val = predict(y_test, x_test, m, "-q")
        ans[0][i] = p_acc[0]

        # Q13: L2 estimate with  Ein
        p_label, p_acc, p_val = predict(y_train, x_train, m, "-q")
        ans[1][i] = p_acc[0]

        # Q18: L1 estimate with Eout
        params = parameter(f"-s 6 -c {1/10**log_lambdas[i]} -e 0.000001 -q")
        m = train(prob, params)
        p_label, p_acc, p_val = predict(y_test, x_test, m, "-q")
        ans[2][i] = p_acc[0]

    print(100-ans)

    argans = np.argmax(ans, axis=1)
    print(log_lambdas[argans])

    # Q19, 20
    mode = [6, 0]  # 2:Q18, 0:Q12
    tmp2 = [1/10**log_lambdas[argans[2]], 1/10**log_lambdas[argans[0]]/2]
    for i in range(2):
        params = parameter(
            f"-s {mode[i]} -c {tmp2[i]} -e 0.000001 -q")
        m = train(prob, params)
        [W, b] = m.get_decfun()
        W = np.array(W)
        ans1920 = np.count_nonzero(abs(W) <= 10**-6)
        print(ans1920)


# Q12, 13, 18
# [[18.8 14.2 15.4 17.8 22.6]
#  [24.   4.   0.   0.   0. ]
#  [50.8 32.  15.4 16.4 24.6]]
# [3. 0. 0.]

# Q19,20
# 960
# 1
