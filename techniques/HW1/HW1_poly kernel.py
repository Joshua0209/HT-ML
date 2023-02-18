from cvxopt import solvers, matrix
import numpy as np


def kernel(x1, x2):
    return (1+np.dot(x1, x2))**2


xs = [[1, 0], [0, 1], [0, -1], [-1, 0], [0, 2], [0, -2], [-2, 0]]
ys = np.array([-1, -1, -1, 1, 1, 1, 1])
Q = np.zeros((7, 7))

for i in range(7):
    for j in range(7):
        Q[i][j] = ys[i]*ys[j]*kernel(np.array(xs[i]), np.array(xs[j]))

p = -np.ones(7)
A = np.zeros((9, 7))
A[0] = ys
A[1] = -ys

for i in range(2, 9):
    A[i][i-2] = -1
c = np.zeros(9)

P = matrix(Q)

q = matrix(p)

G = matrix(A)
h = matrix(c)
solvers.qp()
alphas = solvers.qp(P, q, G, h)
alphas = alphas['x']
print('max alpha:', np.max(alphas))
print('alpha sum:', np.sum(alphas))
print('min alpha:', np.min(alphas))
print('alphas:', alphas)


def kernelParameters(x):
    return np.array([x[0]*x[0], x[1]*x[1], 2*x[0]*x[1], 2*x[0], 2*x[1], 1])


w = np.zeros(6)

for i in range(7):
    w += alphas[i]*ys[i]*kernelParameters(xs[i])
b = ys[1]
for i in range(7):
    b = alphas[i]*ys[i]*kernel(xs[i], xs[1])
print('x1*x1:', w[0], 'x2*x2:', w[1], 'x1*x2:',
      w[2], 'x1:', w[3], 'x2:', w[4], '1:', w[5])
print('b:', b)
