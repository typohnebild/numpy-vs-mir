#!/usr/bin/env python3
import logging

import numpy as np

import multipy.tools.heatmap as hm
import multipy.tools.util as util


logging.basicConfig(level=logging.INFO)

np.set_printoptions(precision=4, linewidth=180)


@util.timer
def run(N, iter=500):
    grid = hm.initMap_2D(N)
    A, U, F = hm.reshape_grid(grid, hm.heat_sources_2D(N))
    U = hm.gauss_seidel(A, F, U, max_iter=iter)
    grid[1:-1, 1:-1] = U.reshape((N - 2, N - 2))
    return grid


@util.timer
def solve(N):
    grid = hm.initMap_2D(N)
    A, U, F = hm.reshape_grid(grid, hm.heat_sources_2D(N))
    U = np.linalg.solve(A, F)
    grid[1:-1, 1:-1] = U.reshape((N - 2, N - 2))
    return grid


@util.timer
def simulate_1D(N, max_iter=500, numba=True):
    U = hm.initMap_1D(N)
    F = hm.heat_sources_1D(N)
    return hm.GS_RB(F, U, h=None, max_iter=max_iter, numba=numba)


@util.timer
def simulate_2D(N, max_iter=20000, numba=True):
    U = hm.initMap_2D(N)
    F = hm.heat_sources_2D(N)
    return hm.GS_RB(-F, U, h=None, max_iter=max_iter, numba=numba)


@util.timer
def simulate_3D(N, max_iter=500, numba=True):
    U = hm.initMap_3D(N)
    F = np.zeros((N, N, N))
    return hm.GS_RB(F, U, max_iter=max_iter, numba=numba)


@util.timer
def simulate_2D_multigrid(N, iter_cycle=5, numba=True):
    U = hm.initMap_2D(N)
    F = hm.heat_sources_2D(N)
    return hm.poisson_multigrid(F, U, 3, 2, 2, 2, iter_cycle, numba=numba)


@util.timer
def simulate_2D_general_multigrid(N, iter_cycle=5):
    grid = hm.initMap_2D(N)
    rhs = hm.heat_sources_2D(N)
    A, U, F = hm.reshape_grid(grid, rhs)
    U = hm.general_multigrid(A, F, U, 3, 2, 2, 2, iter_cycle)
    grid[1:-1, 1:-1] = U.reshape((N - 2, N - 2))
    return grid


@util.timer
def compare():
    N = 10
    max_iter = 500
    h = 1
    grid = hm.initMap_2D(N)
    rhs = hm.heat_sources_2D(N)
    A, U, F = hm.reshape_grid(grid, rhs)
    U1 = hm.GS_RB(rhs, grid.copy(), h=h, max_iter=max_iter)
    U2 = hm.GS_RB(-rhs, grid.copy(), h=h, max_iter=max_iter)

    U3 = hm.gauss_seidel(A, F, U.copy(), max_iter=max_iter)
    U4 = hm.gauss_seidel(A, -F, U.copy(), max_iter=max_iter)

    U5 = np.linalg.solve(A, F.copy())
    U6 = np.linalg.solve(A, -F.copy())
    return U1, U2, U3, U4, U5, U6


def draw2D(U):
    import matplotlib.pyplot as plt
    if len(U.shape) == 1:
        n = int(np.sqrt(U.shape[0]))
        assert n * n == U.shape[0]
        plt.imshow(U.reshape((n, n)), cmap='RdBu_r', interpolation='nearest')
    else:
        plt.imshow(U, cmap='RdBu_r', interpolation='nearest')
    plt.show()


def draw3D(map):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface.
    for index, x in np.ndenumerate(map):
        if x > 0.5:
            ax.scatter(*index, c='black', alpha=max(x - 0.5, 0))

    fig.show()


if __name__ == "__main__":
    simulate_2D_multigrid(40, 10)
