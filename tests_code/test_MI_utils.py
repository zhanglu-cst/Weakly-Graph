import random
import torch
import numpy as np


def calc_MI(X, Y, bins):
    c_XY = np.histogram2d(X, Y, bins)[0]
    c_X = np.histogram(X, bins)[0]
    c_Y = np.histogram(Y, bins)[0]

    H_X = shan_entropy(c_X)
    H_Y = shan_entropy(c_Y)
    H_XY = shan_entropy(c_XY)

    MI = H_X + H_Y - H_XY
    return MI


def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized * np.log2(c_normalized))
    return H


avg = 0
dim_len = 512
for i in range(1000):
    # a = torch.randn(dim_len).unsqueeze(0).softmax(dim = 1)
    # b = torch.randn(dim_len).unsqueeze(0).softmax(dim = 1)
    a = [random.random() * 1 for _ in range(dim_len)]
    b = [random.random() * 1 for _ in range(dim_len)]
    # print(a)
    a = torch.tensor(a).float().numpy()
    b = torch.tensor(b).float().numpy()

    mi = calc_MI(X = a, Y = b, bins = 10)
    aa = calc_MI(X = a, Y = a, bins = 10)
    bb = calc_MI(X = b, Y = b, bins = 10)
    print(mi, aa, bb)
    avg += mi

    # mi2 = calc_MI_2(x = a, y = b, bins = 100)
    # aa = calc_MI_2(x = a, y = a, bins = 100)
    # bb = calc_MI_2(x = b, y = b, bins = 100)
    # print(mi2, aa, bb)
    # print('------\n')
    # res = IIC(a, b, C = dim_len)
    # same = IIC(a, a, C = dim_len)
    # print(res, same)
print(avg / 1000)