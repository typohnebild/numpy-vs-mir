import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, LinearLocator

import Python.tools.heatmap as hm
import numpy as np


def run(dim, iter=500):
    # get initial Heat Map
    U = hm.initMap_2D(dim).flatten()
    F = hm.heat_sources_2D(dim).flatten()
    # apply Gauss Seidel on it
    A = hm.poisson_operator_2D(dim)
    U = hm.gauss_seidel(A, F, U, max_iter=iter)
    # draw result
    return U.reshape((dim, dim))


def simulate_2D(N, max_iter=500):
    U = hm.initMap_2D(N)
    F = hm.heat_sources_2D(N)
    return hm.GS_RB(F, U, h=None, max_iter=max_iter)


def simulate_3D(N, max_iter=500):
    U = hm.initMap_3D(N)
    F = np.zeros((N, N, N))
    return hm.GS_RB(F, U, max_iter=max_iter)


def simulate_2D_multigrid(N):
    U = hm.initMap_2D(N)
    F = hm.heat_sources_2D(N)
    return hm.poisson_multigrid(F, U, 2, 10, 5, 1)


def simulate_2D_gerneral_multigrid(N):
    A = hm.poisson_operator_2D(N)
    U = hm.initMap_2D(N).flatten()
    F = hm.heat_sources_2D(N).flatten()
    U = hm.general_multigrid(A, F, U, int(np.log(N)) - 1, 3, 3, 2)
    return U.reshape((N, N))
