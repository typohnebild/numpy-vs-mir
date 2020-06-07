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
    A[0, 0] = A[-1, -1] = 1
    upper = -1. * np.eye(N, N - 1)
    upper = np.concatenate((np.zeros((N, 1)), upper), axis=1)
    ret = A + upper + upper.T
    ret[0, 1:] = ret[-1, :-1] = 0
    return ret


@timer
def poisson_operator_2D(N):
    """
        return n^2 x n^2 matrix
    """
    B = [poisson_operator(N)] * (N - 2)
    I = np.eye(N)
    middle = block_diag(I, *B, I)
    upper = - np.eye(N * (N - 2), N * (N - 2))
    upper[0, 0] = upper[-1, -1] = 0
    upper = np.pad(upper, ((N, N), (2 * N, 0)))
    return middle + upper + np.flip(upper)
