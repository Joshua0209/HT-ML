import numpy as np
import matplotlib.pyplot as plt


def data(path: str):
    data = np.genfromtxt(path)
    y, X = data[:, -1], data[:, :-1]
    return X, y


class TreeNode:
    '''s: sign; dim: dimension; theta: threshold
    left: left branch; right: right branch'''

    def __init__(self, s: int, dim: int, theta: float):
        self.s = s
        self.dim = dim
        self.theta = theta
        self.left = None
        self.right = None

    def __len__(self):
        '''number of nodes in the tree'''
        if (self.left and self.right) is None:
            return 0
        else:
            res = 1
            res += len(self.left)
            res += len(self.right)
            return res


def giniIndex(y) -> float:
    giniIndex = 1
    for i in np.unique(y):
        giniIndex -= (np.count_nonzero(y == i)/len(y))**2
    return giniIndex


def b(x, y, dim: int, theta: float) -> float:
    '''branching criteria'''
    y1, y2 = y[x[:, dim] < theta], y[x[:, dim] >= theta]
    return len(y1)*giniIndex(y1)+len(y2)*giniIndex(y2)


def stump(x, y):
    best_cri = float("inf")
    for dim in range(len(x[0])):
        thetas = np.sort(x[:, dim])
        thetas = [(thetas[i+1] + thetas[i]) /
                  2 for i in range(len(thetas)-1)]
        for theta in thetas:
            cri = b(x, y, dim, theta)
            if cri < best_cri:
                best_cri = cri
                best_dim, best_theta = dim, theta
    return best_dim, best_theta


def CART(x, y, n: int, N: int):
    '''n: number of branches, N: maximum number of branches'''
    if len(y) == 0:
        return None
    if giniIndex(y) == 0:  # cannot branch anymore
        return TreeNode(y[0], -1, -1)
    else:
        #  learn branching criteria
        dim, theta = stump(x, y)

        # split D to 2 part
        left_x = x[x[:, dim] < theta]
        right_x = x[x[:, dim] >= theta]
        left_y = y[x[:, dim] < theta]
        right_y = y[x[:, dim] >= theta]
        if np.count_nonzero(left_y == -1) > np.count_nonzero(right_y == -1):
            s = 1
        else:
            s = -1
        node = TreeNode(s, dim, theta)
        # build sub-tree
        n += 1
        if n < N:
            node.left = CART(left_x, left_y, n, N)
            node.right = CART(right_x, right_y, n, N)
        return node


def predict(x, Tree: TreeNode):
    if (Tree.left and Tree.right) != None:
        if x[Tree.dim] < Tree.theta:
            return predict(x, Tree.left)
        else:
            return predict(x, Tree.right)
    else:
        if Tree.dim == -1:
            return Tree.s
        else:
            if x[Tree.dim] < Tree.theta:
                return -1*Tree.s
            else:
                return Tree.s


def error(x, y, cri, func) -> float:
    err = 0
    for i in range(len(x)):
        if func(x[i], cri) != y[i]:
            err += 1
    return err/len(y)


if __name__ == "__main__":
    x_train, y_train = data("hw3_dectree_train.dat")
    x_test, y_test = data("hw3_dectree_test.dat")

    Tree = CART(x_train, y_train, 0, float("inf"))
    print(len(Tree))
    print(error(x_train, y_train, Tree, predict))
    print(error(x_test, y_test, Tree, predict))
