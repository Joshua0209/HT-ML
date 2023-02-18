import numpy as np
N = 100000
w, b = 0, 0
for i in range(N):
    x = np.random.random(2)
    y = x**2
    wt = (y[1]-y[0])/(x[1]-x[0])
    bt = y[0]-wt*x[0]
    w += wt
    b += bt
print(w/N, b/N)
