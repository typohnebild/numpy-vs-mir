import numpy as np
import gaussSeidel as gs
from operators import poisson_operator_2D
import matplotlib.pyplot as plt


def initMap_2D(dimension):
    U = np.random.uniform(0, 1, (dimension, dimension))
    U[:, 0] = 1
    U[0, :] = 1
    U[:, -1] = 0
    U[-1, :] = 0
    return U


def heat_sources_2D(dimension):
    F = np.zeros((dimension, dimension))
    F[:, 0] = 1
    F[0, :] = 1
    F[:, -1] = 0
    F[-1, :] = 0
    F[dimension//2, dimension//2] = 1
    return F


def drawMap(map):
    plt.imshow(map, cmap='hot', interpolation='nearest')
    plt.show()


def run(dim, iter=1000):
    # get initial Heat Map
    U = initMap_2D(dim).flatten()
    F = heat_sources_2D(dim).flatten()
    # apply Gauss Seidel on it
    A = poisson_operator_2D(dim)
    U = gs.gauss_seidel(A, F, U, max_iter=iter)
    # draw result
    return U.reshape((dim, dim))


def simulate_2D(N, max_iter=500):
    U = initMap_2D(N)
    F = heat_sources_2D(N)
    return gs.GS_RB(F, U, max_iter=max_iter)
