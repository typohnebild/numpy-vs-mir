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


def drawMap(map):
    plt.imshow(map, cmap='hot', interpolation='nearest')
    plt.show()


def run(dim, iter=1000):
    # get initial Heat Map
    U = initMap_2D(dim).flatten()
    F = np.zeros(dim * dim)
    # apply Gauss Seidel on it
    A = poisson_operator_2D(dim)
    U = gs.gauss_seidel(A, F, U, max_iter=iter)
    # draw result
    return U.reshape((dim, dim))


def simulate_2D(N):
    U = initMap_2D(N)
    F = np.zeros((N, N))
    return gs.GS_RB(F, U)
