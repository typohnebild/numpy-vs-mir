"""
    A example problem
    solves the heat distribution in NxN grid
"""
import numpy as np

from .operators import poisson_operator_2D


def initMap_1D(dimension):
    U = np.random.uniform(0, 1, (dimension))
    U[0] = 0
    U[-1] = 1
    return U


def initMap_2D(dimension):
    np.random.seed(500)
    U = np.random.uniform(0, 1, (dimension, dimension))
    U[:, -1] = 0
    U[-1, :] = 0
    U[:, 0] = 1
    U[0, :] = 1
    return U


def initMap_3D(dimension):
    U = np.random.uniform(0, 1, (dimension, dimension, dimension))
    U[:, -1, :] = 0
    U[-1, :, :] = 0
    U[:, :, -1] = 0
    U[:, 0, :] = 1
    U[0, :, :] = 1
    U[:, :, 0] = 1
    return U


def heat_sources_1D(dimension):
    F = np.zeros((dimension))
    F[0] = 1
    F[1] = 0
    return F


def heat_sources_2D(dimension):
    F = np.zeros((dimension, dimension))
    # F[dimension // 2, dimension // 2] = 1
    F[:, -1] = 0
    F[-1, :] = 0
    F[:, 0] = 1
    F[0, :] = 1
    return F


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


def create_problem_2D(N):
    return initMap_2D(N), heat_sources_2D(N)
