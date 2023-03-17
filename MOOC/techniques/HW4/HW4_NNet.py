import numpy as np
import torch


def data(path: str):
    data = np.genfromtxt(path)
    y, X = data[:, -1], data[:, :-1]
    return torch.from_numpy(X), torch.from_numpy(y)


def tanhp(x: float) -> float:
    return 1-torch.tanh(x)**2


class NeuralNetwork():
    '''d-M-1 network with all tanh neurons connected'''

    def __init__(self, x_train, y_train, M: list = [3], r: float = 0.1, eta: float = 0.1):
        self.x_train = x_train
        self.y_train = y_train
        self.layer = len(M)  # number of hidden layer
        self.din = len(x_train[0])
        try:
            self.dout = len(y_train[1])
        except TypeError:
            self.dout = 1
        self.M = M
        self.r = r
        self.eta = eta

        self.S = [0]*(self.layer+1)  # W-S-tanh
        self.X = [0]*(self.layer+1)  # tanh-X-W
        self.Delta = [0]*(self.layer+1)
        tmp = M + [self.dout]
        for l in range(self.layer + 1):
            self.S[l] = torch.zeros(tmp[l])
            self.X[l] = torch.zeros(tmp[l])
            self.Delta[l] = torch.zeros(tmp[l])

        self.W = [0]*(self.layer+1)
        tmp = [self.din]+M+[self.dout]
        for l in range(self.layer+1):
            self.W[l] = torch.FloatTensor(tmp[l]+1, tmp[l+1]).uniform_(-r, r)

    def forward(self, x0):
        for l in range(self.layer+1):
            tmp = [x0]+self.X
            tmp2 = torch.cat((torch.ones(1), tmp[l]), dim=0)
            self.S[l] = self.W[l].T@tmp2.float()
            self.X[l] = torch.tanh(self.S[l])

    def backward(self, y0):
        for l in range(self.layer, -1, -1):
            if l == self.layer:
                self.Delta[l] = -2*(y0-np.tanh(self.S[l]))@tanhp(self.S[l])
                self.Delta[l] = self.Delta[l].unsqueeze(0)
            else:
                self.Delta[l] = self.W[l+1][1:, :]@self.Delta[l+1] \
                    * tanhp(self.S[l])

    def grad_des(self, x0):
        tmp = [torch.mean(x0, axis=0)]+self.X
        for l in range(len(self.W)):
            tmp_x = torch.cat(
                (torch.ones(1), tmp[l]), dim=0).float().unsqueeze(0)
            self.W[l] -= self.eta*tmp_x.T@self.Delta[l].unsqueeze(0)

    def backprop(self, T: int, n: int):
        '''T: iteration; n: mini batch size'''
        for _ in range(T):
            # stochastic
            idx = torch.randint(0, len(self.x_train), size=(n,))
            x_tmp, y_tmp = self.x_train[idx], self.y_train[idx]
            for i in range(n):
                self.forward(x_tmp[i])
                self.backward(y_tmp[i])
            self.grad_des(x_tmp)


def error(x, y):
    yhat = torch.zeros_like(y)
    for i in range(len(x)):
        nn.forward(x[i])
        yhat[i] = nn.S[-1]
    return float(torch.count_nonzero(torch.sign(yhat) != y)/len(y))


if __name__ == "__main__":
    x_train, y_train = data("MOOC/techniques/HW4/hw4_nnet_train.dat")
    x_test, y_test = data("MOOC/techniques/HW4/hw4_nnet_test.dat")
    N = 1
    T, n = 50000, 1
    Ms = [[1], [6], [11], [16], [21]]  # 6
    rs = [0, 0.001, 0.1, 10, 1000]  # 0.001
    etas = [0.001, 0.01, 0.1, 1, 10]  # 0.01

    err_Ms = [0, 0, 0, 0, 0]
    err_rs = [0, 0, 0, 0, 0]
    err_etas = [0, 0, 0, 0, 0]
    err_83 = 0

    for _ in range(N):
        for i in range(len(Ms)):
            nn = NeuralNetwork(x_train, y_train, M=Ms[i])
            nn.backprop(T, n)
            err_Ms[i] += error(x_test, y_test)
        print("Ms finished")

        for i in range(len(rs)):
            nn = NeuralNetwork(x_train, y_train, r=rs[i])
            nn.backprop(T, n)
            err_rs[i] += error(x_test, y_test)
        print("rs finished")

        for i in range(len(etas)):
            nn = NeuralNetwork(x_train, y_train, eta=etas[i])
            nn.backprop(T, n)
            err_etas[i] += error(x_test, y_test)
        print("etas finished")

        nn = NeuralNetwork(x_train, y_train)
        nn = NeuralNetwork(x_train, y_train, M=[8, 3], eta=0.01)
        nn.backprop(T, n)
        err_83 += error(x_test, y_test)

    print(err_Ms, Ms[np.argmin(np.array(err_Ms))])
    print(err_rs, rs[np.argmin(np.array(err_rs))])
    print(err_etas, etas[np.argmin(np.array(err_etas))])
    print(err_83/N)  # 0.036

# [0.472, 0.036, 0.036, 0.24, 0.04][6]
# [0.472, 0.036, 0.036, 0.472, 0.452] 0.001
# [0.076, 0.036, 0.036, 0.428, 0.528] 0.01
