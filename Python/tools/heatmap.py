import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter, LinearLocator

from ..GaussSeidel.GaussSeidel import gauss_seidel
from ..GaussSeidel.GaussSeidel_RB import GS_RB
from ..multigrid import poisson_multigrid, general_multigrid
from .operators import poisson_operator_2D


def initMap_2D(dimension):
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


def heat_sources_2D(dimension):
    F = np.zeros((dimension, dimension))
    F[:, -1] = 0
    F[-1, :] = 0
    F[:, 0] = 1
    F[0, :] = 1
    F[dimension // 2, dimension // 2] = 1
    return F


def run(dim, iter=500):
    # get initial Heat Map
    U = initMap_2D(dim).flatten()
    F = heat_sources_2D(dim).flatten()
    # apply Gauss Seidel on it
    A = poisson_operator_2D(dim)
    U = gauss_seidel(A, F, U, max_iter=iter)
    # draw result
    return U.reshape((dim, dim))


def simulate_2D(N, max_iter=500):
    U = initMap_2D(N)
    F = heat_sources_2D(N)
    return GS_RB(F, U, h=None, max_iter=max_iter)


def simulate_3D(N, max_iter=500):
    U = initMap_3D(N)
    F = np.zeros((N, N, N))
    return GS_RB(F, U, max_iter=max_iter)


def simulate_2D_multigrid(N):
    U = initMap_2D(N)
    F = heat_sources_2D(N)
    return poisson_multigrid(F, U, 2, 10, 5, 1)


def simulate_2D_gerneral_multigrid(N):
    A = poisson_operator_2D(N)
    U = initMap_2D(N).flatten()
    F = heat_sources_2D(N).flatten()
    U = general_multigrid(A, F, U, int(np.log(N)) - 1, 3, 3, 2)
    return U.reshape((N, N))
