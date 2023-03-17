
import numpy as np


def pfunc(x):
    u, v = x
    return np.array([np.exp(u) + v*np.exp(u*v) + 2*u - 2*v - 3, 2*np.exp(2*v) + u*np.exp(u*v) - 2*u + 4*v - 2])


def func(x):
    u, v = x
    return np.exp(u)+np.exp(2*v) + np.exp(u*v) + u**2 - 2*u*v + 2*v**2 - 3*u - 2*v


def ppfunc(x):
    u, v = x
    uu = np.exp(u) + v**2*np.exp(u*v) + 2
    uv = (1+u*v)*np.exp(u*v) - 2
    vv = 4*np.exp(2*v) + u**2*np.exp(u*v) + 4
    return np.array([[uu, uv], [uv, vv]])


x = np.array([0, 0])
for i in range(5):
    x = x - np.matmul(np.linalg.inv(ppfunc(x)), pfunc(x))
print(x, func(x))
