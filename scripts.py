import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, LinearLocator

import Python.tools.heatmap as hm
import numpy as np
import Python.tools.util as util

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
def simulate_1D(N, max_iter=500):
    U = hm.initMap_1D(N)
    F = hm.heat_sources_1D(N)
    return hm.GS_RB(F, U, h=None, max_iter=max_iter)


@util.timer
def simulate_2D(N, max_iter=20000):
    U = hm.initMap_2D(N)
    F = hm.heat_sources_2D(N)
    return hm.GS_RB(-F, U, h=None, max_iter=max_iter)


@util.timer
def simulate_3D(N, max_iter=500):
    U = hm.initMap_3D(N)
    F = np.zeros((N, N, N))
    return hm.GS_RB(F, U, max_iter=max_iter)


@util.timer
def simulate_2D_multigrid(N, iter_cycle=5):
    U = hm.initMap_2D(N)
    F = hm.heat_sources_2D(N)
    return hm.poisson_multigrid(F, U, 5, 2, 2, 2, iter_cycle)


@util.timer
def simulate_2D_general_multigrid(N):
    grid = hm.initMap_2D(N)
    rhs = hm.heat_sources_2D(N)
    A, U, F = hm.reshape_grid(grid, rhs)
    U = hm.general_multigrid(A, F, U, 2, 5, 5, 1)
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
