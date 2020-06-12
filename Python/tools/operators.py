import numpy as np
from scipy.linalg import block_diag

from .util import timer


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


def poisson_operator(N, h):
    """
        returns a Matrix with  nxn -1 4 -1 on diagonal
        @param h is distance between grid points
    """
    A = 4. * np.eye(N, N) / (h * h)
    A[0, 0] = A[-1, -1] = 1
    upper = -1. * np.eye(N, N - 1) / (h * h)
    upper = np.concatenate((np.zeros((N, 1)), upper), axis=1)
    ret = A + upper + upper.T
    ret[0, 1:] = ret[-1, :-1] = 0
    return ret


# @timer
def poisson_operator_2D(N, h=None):
    """
        return n^2 x n^2 matrix
        @param h is distance between grid points
    """
    if h is None:
        h = 1/N

    B = [poisson_operator(N,h)] * (N - 2)
    I = np.eye(N)
    middle = block_diag(I, *B, I)
    upper = - np.eye(N, N) / (h * h)
    upper[0, 0] = upper[-1, -1] = 0
    upper = block_diag(*[upper] * (N - 2))
    upper = np.pad(upper, ((N, N), (2 * N, 0)))
    return middle + upper + np.flip(upper)
