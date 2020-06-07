import numpy as np
import gaussSeidel as gs
from operators import poisson_operator_2D
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator, FormatStrFormatter


def initMap_2D(dimension):
    U = np.random.uniform(0, 1, (dimension, dimension))
    U[:, 0] = 1
    U[0, :] = 1
    U[:, -1] = 0
    U[-1, :] = 0
    return U


def initMap_3D(dimension):
    U = np.random.uniform(0, 1, (dimension, dimension, dimension))
    U[:, 0, :] = 1
    U[0, :, :] = 1
    U[:, :, 0] = 1
    U[:, -1, :] = 0
    U[-1, :, :] = 0
    U[:, :, -1] = 0
    return U


def heat_sources_2D(dimension):
    F = np.zeros((dimension, dimension))
    F[:, 0] = 1
    F[0, :] = 1
    F[:, -1] = 0
    F[-1, :] = 0
    F[dimension//2, dimension//2] = 1
    return F


def draw2D(map):
    plt.imshow(map, cmap='RdBu_r', interpolation='nearest')
    plt.show()


def draw3D(map):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface.
    for index, x in np.ndenumerate(map):
        if x > 0.5:
            ax.scatter(*index, c='black', alpha=max(x - 0.5, 0))

    fig.show()


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


def simulate_3D(N, max_iter=500):
    U = initMap_3D(N)
    F = np.zeros((N, N, N))
    return gs.GS_RB(F, U, max_iter=max_iter)
