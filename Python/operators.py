import numpy as np


def restriction_operator(N):
    """
        should return the restriction operator matrix from R^(N-1) -> R^(N/2-1)
    """
    diag = np.array([1 / 4, 1 / 2, 1 / 4])
    zeros = np.zeros(N - 2)
    conc = np.concatenate((diag, zeros))
    ret = np.tile(conc, N // 2 - 2)
    ret = np.concatenate((ret, diag))
    return ret.reshape((N // 2 - 1, N - 1))
