import numpy as np

path = "2023_spring\HW1\hw1_train.dat"
data = np.genfromtxt(path)
y, X_data = data[:, -1], data[:, :-1]
tmp = np.ones(len(X_data))
X = np.vstack((tmp, X_data.T)).T
N = len(X)


def error(w):
    return np.count_nonzero((X@w)*y < 0)/len(X)


def PLA(M):
    err = 0.
    n = 1000
    for i in range(n):
        w = np.zeros_like(X[0])
        m = 0
        while m < M:
            idx = np.random.randint(0, len(X))
            if (w@X[idx]).sum()*y[idx] <= 0:
                w += X[idx]*y[idx]
                m = 0
            else:
                m += 1

        err += error(w)

    return err/n


if __name__ == '__main__':
    ans13 = PLA(int(N//2))
    print(ans13)
    ans14 = PLA(int(4*N))
    print(ans14)

# 0.0202734375
# 0.0001875

# 0.02040234375
# 0.00022265625
