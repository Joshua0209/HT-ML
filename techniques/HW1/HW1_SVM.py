from SVM import SVM
import numpy as np

train_data_path = "train.dat"
test_data_path = "test.dat"
# Q15
s = SVM(train_data_path, test_data_path, digit=0)
print(np.linalg.norm(s.train(kernel='linear', C=0.01).coef_))

# Q16, 17
digits = [0, 2, 4, 6, 8]
errorin = []
alpha_sum = []
for digit in digits:
    s = SVM(train_data_path, test_data_path, digit=digit)
    s.train(kernel='poly', C=0.01, degree=2)
    errorin.append(s.calculate_error("Ein"))
    alpha_sum.append(np.sum(np.fabs(s.model.dual_coef_[0])))
print(errorin)
print(alpha_sum)

# Q18
logCs = [-3, -2, -1, 0, 1]
errout = []
nsupport = []
s = SVM(train_data_path, test_data_path, digit=0)
for logC in logCs:
    s.train(kernel='rbf', C=10**logC)
    errout.append(s.calculate_error("Eout"))
    nsupport.append(len(s.model.support_vectors_))
print(nsupport)
print(errout)

# Q19
logGammas = [0, 1, 2, 3, 4]
errout = []
s = SVM(train_data_path, test_data_path, digit=0)
for logGamma in logGammas:
    s.train(kernel='rbf', C=0.01, gamma=10**logGamma)
    errout.append(s.calculate_error("Eout"))
print(errout)

# Q20
logGammas = [0, 1, 2, 3, 4]
s = SVM(train_data_path, test_data_path, digit=0)
N = 3
# count = np.zeros(5)
for _ in range(N):
    s.train_valid_split(1000, digit=0)
    errval = float("inf")
    Eval = []
    for logGamma in logGammas:
        s.train(kernel='rbf', C=0.1, gamma=10**logGamma)
        Eval.append(s.calculate_error("Eval"))
print(Eval)
