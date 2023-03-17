from numpy import sqrt, log
from scipy.optimize import fsolve
N = 5
d = 50.
delta = 0.05

ep1 = sqrt(8/N*log(4*(2*N)**d/delta))
ep2 = sqrt(2*log(2*N*(N)**d)/N) + sqrt(2/N*log(1/delta)) + 1/N


def ep3(x):
    return sqrt(1/N*(2*x+log(6*(2*N)**d/delta)))-x


def ep4(x):
    return sqrt((4*x*(1+x)+2*d*log(N)+log(4 / delta))/(2*N))-x


ep5 = sqrt(16/N*log(2*N**d/sqrt(delta)))
print(ep1, ep2, fsolve(ep3, 1)[0], fsolve(ep4, 1)[0], ep5)
