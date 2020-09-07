"""
    A example problem
    solves the heat distribution in NxN grid
"""
import numpy as np


def initMap_1D(N):
    U = np.random.uniform(0, 1, (N))
    U[0] = 0
    U[-1] = 1
    return U


def initMap_2D(N):
    U = np.random.uniform(0, 1, (N, N))
    U[:, -1] = 0
    U[-1, :] = 0
    U[:, 0] = 1
    U[0, :] = 1
    return U


def initMap_3D(N):
    U = np.random.uniform(0, 1, (N, N, N))
    U[:, -1, :] = 0
    U[-1, :, :] = 0
    U[:, :, -1] = 0
    U[:, 0, :] = 1
    U[0, :, :] = 1
    U[:, :, 0] = 1
    return U


def heat_sources_1D(N):
    F = np.zeros((N))
    F[0] = 1
    F[1] = 0
    return F


def heat_sources_2D(N):
    F = np.zeros((N, N))
    F[:, -1] = 0
    F[-1, :] = 0
    F[:, 0] = 1
    F[0, :] = 1
    return F


def heat_sources_3D(N):
    F = np.zeros((N, N, N))
    F[:, -1, :] = 0
    F[-1, :, :] = 0
    F[:, :, -1] = 0
    F[:, 0, :] = 1
    F[0, :, :] = 1
    F[:, :, 0] = 1
    return F


def create_problem_1D(N):
    return initMap_1D(N), heat_sources_1D(N)


def create_problem_2D(N):
    return initMap_2D(N), heat_sources_2D(N)


def create_problem_3D(N):
    return initMap_3D(N), heat_sources_3D(N)
