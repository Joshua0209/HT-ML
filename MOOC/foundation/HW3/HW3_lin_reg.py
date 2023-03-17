import numpy as np
import matplotlib.pyplot as plt


def data(N):
    x1, x2 = np.random.uniform(-1, 1, N), np.random.uniform(-1, 1, N)
    # x = np.vstack((x1, x2)).T
    x = np.vstack((np.ones(N), x1, x2, x1*x2, x1**2, x2**2)).T
    y = np.sign(x1**2+x2**2-0.6)
    noise = np.random.uniform(0, 1, N)
    y[noise < 0.1] *= -1
    return x, y


n = 1000
err = 0
for i in range(n):
    N = 1000
    x, y = data(N)
    hat = np.matmul(np.linalg.inv(np.matmul(x.T, x)), x.T)
    w = np.matmul(hat, y)
    x, y = data(N)
    err += np.count_nonzero(y != np.sign(np.inner(w, x)))
print(err/n/N)

# plt.figure(figsize=(6, 6))
# plt.scatter(x1[y > 0], x2[y > 0], color='blue')
# plt.scatter(x1[y < 0], x2[y < 0], color='red')
# ppi = np.linspace(0, 2*np.pi, 100)
# plt.scatter(np.sqrt(0.6)*np.sin(ppi), 0.6*np.cos(ppi))
# px = np.linspace(-1, 1, 100)
# plt.scatter(px, w[0]*px+w[1])
# plt.show()
