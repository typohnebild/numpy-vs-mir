import numpy as np
from scipy.linalg import block_diag
from util import timer


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


def poisson_operator(N):
    """
        returns a Matrix with  nxn -1 4 -1 on diagonal
    """
    A = 4. * np.eye(N, N)
    upper = -1. * np.eye(N, N - 1)
    upper = np.concatenate((np.zeros((N, 1)), upper), axis=1)
    return A + upper + upper.T


@timer
def poisson_operator_2D(N):
    """
        return n^2 x n^2 matrix
    """
    B = poisson_operator(N)
    # I = -np.eye(N)
    middle = block_diag(*[B] * N)
    upper = - np.eye(N * N, N * (N - 1))
    upper = np.concatenate((np.zeros((N * N, N)), upper), axis=1)
    return middle + upper + upper.T