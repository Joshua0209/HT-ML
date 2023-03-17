import numpy as np

path = "2023_spring\HW1\hw1_train.dat"
data = np.genfromtxt(path)
y, X_data = data[:, -1], data[:, :-1]


def PLA(x0, scaling=False):
    if scaling:
        tmp = np.ones(len(X_data))
        X = np.vstack((tmp, X_data.T)).T
        X *= x0
    else:
        tmp = np.ones(len(X_data))*x0
        X = np.vstack((tmp, X_data.T)).T
    N = len(X)
    n = 1000
    M = 4*N
    list_update = []
    list_w0 = []
    for i in range(n):
        w = np.zeros_like(X[0])
        m = 0
        update = 0
        while m < M:
            idx = np.random.randint(0, len(X))
            if (w@X[idx]).sum()*y[idx] <= 0:
                w += X[idx]*y[idx]
                update += 1
                m = 0
            else:
                m += 1

        list_update.append(update)
        list_w0.append(w[0]*x0)

    median_update = np.median(np.array(list_update))
    median_w0 = np.median(np.array(list_w0))
    return median_update, median_w0


if __name__ == "__main__":
    ans15, ans16 = PLA(x0=1)
    print(ans15, ans16)
    ans17, tmp = PLA(x0=0.5, scaling=True)
    print(ans17)
    ans18, tmp = PLA(x0=0)
    print(ans18)
    tmp, ans19 = PLA(x0=-1)
    print(ans19)
    tmp, ans20 = PLA(x0=0.1126)
    print(ans20)

# 453.0 34.0
# 450.0
# 447.0
# 34.0
# 0.4310778400000001
