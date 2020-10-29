#!/usr/bin/env python3
import logging

import numpy as np

import problemgenerator.heatmap as hm
import problemgenerator.femwave as fw
import multipy.tools.operators as op
import multipy.tools.util as util
from multipy.multigrid import poisson_multigrid
from multipy.GaussSeidel import GaussSeidel as gs
from multipy.GaussSeidel import GaussSeidel_RB as gsrb


logging.basicConfig(level=logging.INFO)
logging.getLogger('multipy.multigrid').setLevel(level=logging.DEBUG)
np.set_printoptions(precision=4, linewidth=180)


@util.timer
def run(N, iter=500):
    grid = hm.initMap_2D(N)
    A, U, F = op.reshape_grid(grid, hm.heat_sources_2D(N))
    U = gs.gauss_seidel(A, F, U, max_iter=iter)
    grid[1:-1, 1:-1] = U.reshape((N - 2, N - 2))
    return grid


@util.timer
def solve(N):
    grid = hm.initMap_2D(N)
    A, U, F = op.reshape_grid(grid, hm.heat_sources_2D(N))
    U = np.linalg.solve(A, F)
    grid[1:-1, 1:-1] = U.reshape((N - 2, N - 2))
    return grid


@util.timer
def simulate_1D(N, max_iter=500):
    U = hm.initMap_1D(N)
    F = hm.heat_sources_1D(N)
    return gsrb.GS_RB(F, U, h=None, max_iter=max_iter)


@util.timer
def simulate_2D(N, max_iter=20000):
    U = hm.initMap_2D(N)
    F = hm.heat_sources_2D(N)
    return gsrb.GS_RB(F, U, h=None, max_iter=max_iter)


@util.timer
def simulate_3D(N, max_iter=500):
    U = hm.initMap_3D(N)
    F = np.zeros((N, N, N))
    return gsrb.GS_RB(F, U, max_iter=max_iter)


@util.timer
def simulate_2D_multigrid(N, iter_cycle=5):
    U = hm.initMap_2D(N)
    F = hm.heat_sources_2D(N)
    return poisson_multigrid(F, U, 0, 2, 2, 2, iter_cycle)

@util.timer
def simulate_2D_FEM_multigrid(N, iter_cycle=5):
    U, F = fw.create_2D(N)
    h = 1/(N - 1)
    return poisson_multigrid(F, U, 0, 2, 2, 1, iter_cycle, h)


@util.timer
def simulate_3D_multigrid(N, iter_cycle=5):
    U = hm.initMap_3D(N)
    F = hm.heat_sources_3D(N)
    return poisson_multigrid(F, U, 0, 2, 2, 2, iter_cycle)


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


# if __name__ == "__main__":
#     simulate_2D_multigrid(40, 10)
