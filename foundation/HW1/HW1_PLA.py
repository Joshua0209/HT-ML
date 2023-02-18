import requests
import random

response = requests.get(
    'https://www.csie.ntu.edu.tw/~htlin/mooc/datasets/mlfound_math/hw1_15_train.dat')
data = response.text
data = data.split('\n')
x = list()
data.pop(-1)
for i in data:
    # print(i.split())
    x.append([1]+i.split())

for i in range(len(x)):
    for j in range(len(x[0])):
        x[i][j] = float(x[i][j])


halt = 0
for i in range(2000):

    random.shuffle(x)
    w = [0, 0, 0, 0, 0]

    def dot(a: list, b: list) -> float:
        out = 0
        for i in range(len(a)):
            out += a[i]*b[i]
        return out

    for i in range(len(x)):
        if dot(w, x[i][:-1])*x[i][-1] <= 0:
            for j in range(len(w)):
                w[j] += x[i][j]*x[i][-1]
            halt += 1

print(halt/2000)
