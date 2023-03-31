import numpy as np


def data(tau: float, size: int) -> tuple[float]:
    X = np.random.uniform(-0.5, 0.5, size=size)
    noise = np.random.uniform(0, 1, size)
    y = np.sign(X)
    y[noise < tau] *= -1
    return X, y


def error(t: float, s: int, x, y) -> float:
    fx = s*np.where(x > t, 1, -1)
    return np.count_nonzero(fx != y)/len(y)


def min_err(x, y) -> float:
    tmp = np.sort(x)
    thetas = [(tmp[i+1]+tmp[i])/2 for i in range(len(tmp)-1)]+[float("-inf")]
    min_err = float("inf")
    for theta in thetas:
        for s in [-1, 1]:
            err = error(theta, s, x, y)
            if err < min_err:
                min_err = err
                min_theta = theta
                min_s = s
    return min_err, min_theta, min_s


def main(N: int, tau: float, size: int) -> float:
    Eout_analytic, Ein, Eout_test = [], [], []
    for _ in range(N):
        x_train, y_train = data(tau, size)
        best_err, best_theta, best_s = min_err(x_train, y_train)
        Ein.append(best_err)
        errout = min(abs(best_theta), 0.5)*(1-2*tau) + tau
        Eout_analytic.append(errout)
        x_test, y_test = data(tau, size)
        Eout_test.append(error(best_theta, best_s, x_test, y_test))
    return np.mean(np.array(Eout_analytic) - np.array(Ein)), np.mean(np.array(Eout_test)-np.array(Ein))


if __name__ == '__main__':
    N = 100000
    ans13 = main(N=N, tau=0, size=2)
    print(ans13)
    ans14 = main(N=N, tau=0, size=128)
    print(ans14)
    ans15 = main(N=N, tau=0.20, size=2)
    print(ans15)
    ans16 = main(N=N, tau=0.20, size=128)
    print(ans16)

# (0.29182612146373565, 0.292025)
# (0.0038744481259042037, 0.00388765625)
# (0.3912153904392298, 0.423875)
# (0.014249643693964335, 0.014182890625)
