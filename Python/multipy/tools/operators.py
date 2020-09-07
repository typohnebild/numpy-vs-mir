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
    A = 4. * np.eye(N, N)
    upper = -1. * np.eye(N, N - 1)
    upper = np.concatenate((np.zeros((N, 1)), upper), axis=1)
    ret = A + upper + upper.T
    return ret


def poisson_operator_2D(N, h=None):
    """
        return n^2 x n^2 matrix
        @param h is distance between grid points
    """
    if h is None:
        h = 1 / N

    B = poisson_operator(N, h)
    middle = block_diag(*[B] * N)
    upper = - np.eye(N * N, N * (N - 1))
    upper = np.concatenate((np.zeros((N * N, N)), upper), axis=1)
    return middle + upper + upper.T


def poisson_operator_like(x):
    assert len(x.shape) == 1
    N = x.shape[0]
    ret = 4 * np.eye(N, N)
    ret[0, 1] = ret[-1, -2] = - 1
    if 3 < N:
        ret[0, 3] = ret[-1, -4] = -1
    for i in range(1, N - 1):
        ret[i, i + 1] = - 1
        ret[i, i - 1] = - 1
        if 3 <= i:
            ret[i, i - 3] = - 1
        if i < N - 3:
            ret[i, i + 3] = - 1

    return ret


def boundary_condition(U):
    N = U.shape[0] - 2
    ret = np.zeros(N ** 2)
    # left boundary
    ret[:N] = U[1:-1, 0]
    # right boundary
    ret[-N:] = U[-1, 1:-1]

    # Top boundary
    ret[::N] += U[0, 1:-1:]

    # bottom boundary
    ret[N - 1::N] += U[-1, 1:-1:]
    return ret


def reshape_grid(grid, rhs, h=None):
    """
        Takes a grid and a rhs and reformulates it to
        AU = F with A as poisson operator
        @param h is the distance between the grid points
    """
    assert grid.shape == rhs.shape
    N = grid.shape[0]
    if h is None:
        h = 1 / N
    A = poisson_operator_2D(N - 2)
    U = grid[1:-1, 1:-1].flatten()
    F = h * h * rhs[1:-1, 1:-1].flatten() + boundary_condition(grid)
    return A, U, F
