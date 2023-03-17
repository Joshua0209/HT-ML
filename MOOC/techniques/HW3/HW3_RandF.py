from HW3_CART import *


def boostrap(x, y, N: int):
    m, n = x.shape
    idx = np.random.randint(0, m, N)
    return x[idx], y[idx]


def randF(x, y, T: int, N):
    forest = []
    err = 0
    for t in range(T):
        x_temp, y_temp = boostrap(x, y, len(x))
        forest.append(CART(x_temp, y_temp, 0, N))
        err += error(x, y, forest[t], predict)
    return forest, err/T


def predict_randF(x, forest):
    vote = [0, 0]
    for i in range(len(forest)):
        if predict(x, forest[i]) == 1:
            vote[1] += 1
        else:
            vote[0] += 1
    if np.argmax(vote) == 0:
        return -1
    else:
        return 1


if __name__ == "__main__":
    T = 300  # number of trees in each forest
    N = 10  # number of experiments
    x_train, y_train = data("MOOC/techniques/HW3/hw3_dectree_train.dat")
    x_test, y_test = data("MOOC/techniques/HW3/hw3_dectree_test.dat")
    tot_err_gt = 0
    tot_Ein = 0
    tot_Eout = 0
    for n in range(N):
        # forest, err_gt = randF(x_train, y_train, T, float("inf"))
        forest, err_gt = randF(x_train, y_train, T, 1)
        tot_err_gt += err_gt
        tot_Ein += error(x_train, y_train, forest, predict_randF)
        tot_Eout += error(x_test, y_test, forest, predict_randF)
    print(tot_err_gt/N)
    print(tot_Ein/N)
    print(tot_Eout/N)

# 0.052031999999999946
# 0.0
# 0.07548000000000002
