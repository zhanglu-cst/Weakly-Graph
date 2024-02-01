import numpy as np
import torch


def calc_MI(X, Y, bins = 10):
    if (isinstance(X, torch.Tensor)):
        X = X.squeeze()
        Y = Y.squeeze()
        assert X.dim() == 1 and Y.dim() == 1
        X = X.cpu().clone().detach().numpy()
        Y = Y.cpu().clone().detach().numpy()

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
