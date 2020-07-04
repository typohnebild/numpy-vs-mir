import logging

import numpy as np

import multipy.tools.heatmap as hm
import multipy.tools.util as util


logging.basicConfig(level=logging.INFO)

np.set_printoptions(precision=4, linewidth=180)


@util.profiling
def profile_2D_multigrid(N):
    iter_cycle = 1000
    U = hm.initMap_2D(N)
    F = hm.heat_sources_2D(N)
    hm.poisson_multigrid(F, U, 5, 2, 2, 2, iter_cycle)


@util.profiling
def profile_2D_general_multigrid():
    N = 100
    grid = hm.initMap_2D(N)
    rhs = hm.heat_sources_2D(N)
    A, U, F = hm.reshape_grid(grid, rhs)
    U = hm.general_multigrid(A, F, U, 2, 5, 5, 1)
    grid[1:-1, 1:-1] = U.reshape((N - 2, N - 2))


if __name__ == "__main__":
    for i in range(8, 12):
        profile_2D_multigrid(2**i)
