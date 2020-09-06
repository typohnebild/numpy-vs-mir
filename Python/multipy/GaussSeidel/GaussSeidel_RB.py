import logging

import numpy as np
from numba import jit

from ..tools.apply_poisson import apply_poisson
from ..tools.util import timer

logger = logging.getLogger('GS')
logger.setLevel(logging.WARNING)


# @timer
def GS_RB(
        F,
        U=None,
        h=None,
        max_iter=10_000_000,
        eps=1e-8,
        norm_iter=1000,
        numba=True):
    """Implementation of Gauss Seidl Red Black iterations
       should solve AU = F
       A poisson equation
       @param F n vector
       @param h is distance between grid points | default is 1/N
       @return U n vector
    """

    if U is None:
        U = np.zeros_like(F)
    if h is None:
        h = 1 / (U.shape[0])

    h2 = h * h

    if len(F.shape) == 1:
        # do the sweep
        sweep = sweep_1D if numba else sweep_1D.py_func
    elif len(F.shape) == 2:
        # do the sweep
        sweep = sweep_2D if numba else sweep_2D.py_func
    elif len(F.shape) == 3:
        # Anzahl an Gauss-Seidel-Iterationen ausfuehren
        sweep = sweep_3D if numba else sweep_3D.py_func
    else:
        raise ValueError("Wrong Shape!!!")

    # Anzahl an Gauss-Seidel-Iterationen ausfuehren
    for it in range(1, max_iter + 1):
        # check sometimes if solutions converges
        if it % norm_iter == 0:
            r = F - apply_poisson(U, h)
            norm = np.linalg.norm(r[1:-1, 1:-1])
            if norm <= eps:
                logger.info(
                    f"converged after {it} iterations with {norm:.4}")
                break

        # rote Halbiteration
        sweep(1, F, U, h2)
        # schwarze Halbiteration
        sweep(0, F, U, h2)

    return U


# --- 1D Fall ---
@jit(nopython=True, fastmath=True)
def sweep_1D(color, F, U, h2):
    """
    Does the sweeps
    @param color 1 = red 0 for black
    @param h2 is distance between grid points squared
    """
    n = F.shape[0]
    if color == 1:
        # red
        U[1:n - 1:2] = (U[0:n - 2:2] + U[2::2] - F[1:n - 1:2] * h2) / (2.0)
    else:
        # black
        U[2:n - 1:2] = (U[1:n - 2:2] + U[3::2] - F[2:n - 1:2] * h2) / (2.0)


@jit(nopython=True, fastmath=True)
def sweep_2D(color, F, U, h2):
    """
    Does the sweeps
    @param color 1 = red 0 for black
    @param h2 is distance between grid points squared
    """

    m, n = F.shape

    if color == 1:
        # red
        U[1:m - 1:2, 2:n - 1:2] = (U[0:m - 2:2, 2:n - 1:2] +
                                   U[2::2, 2:n - 1:2] +
                                   U[1:m - 1:2, 1:n - 2:2] +
                                   U[1:m - 1:2, 3::2] -
                                   F[1:m - 1:2, 2:n - 1:2] * h2) / (4.0)
        U[2:m - 1:2, 1:n - 1:2] = (U[1:m - 2:2, 1:n - 1:2] +
                                   U[3::2, 1:n - 1:2] +
                                   U[2:m - 1:2, 0:n - 2:2] +
                                   U[2:m - 1:2, 2::2] -
                                   F[2:m - 1:2, 1:n - 1:2] * h2) / (4.0)
    else:
        # black
        U[1:m - 1:2, 1:n - 1:2] = (U[0:m - 2:2, 1:n - 1:2] +
                                   U[2::2, 1:n - 1:2] +
                                   U[1:m - 1:2, 0:n - 2:2] +
                                   U[1:m - 1:2, 2::2] -
                                   F[1:m - 1:2, 1:n - 1:2] * h2) / (4.0)
        U[2:m - 1:2, 2:n - 1:2] = (U[1:m - 2:2, 2:n - 1:2] +
                                   U[3::2, 2:n - 1:2] +
                                   U[2:m - 1:2, 1:n - 2:2] +
                                   U[2:m - 1:2, 3::2] -
                                   F[2:m - 1:2, 2:n - 1:2] * h2) / (4.0)
# ----------------


# --- 3D Fall ---
@jit(nopython=True, fastmath=True)
def sweep_3D(color, F, U, h):
    """
    Does the sweeps
    @param color 1 = red 0 for black
    @param h is distance between grid points
    """

    m, n, o = F.shape

    for k in range(1, o - 1):
        for j in range(1, n - 1):
            for i in range(1, m - 1):
                if (i + j + k) % 2 == color:
                    U[i, j, k] = (U[i - 1, j, k] +
                                  U[i + 1, j, k] +
                                  U[i, j - 1, k] +
                                  U[i, j + 1, k] +
                                  U[i, j, k - 1] +
                                  U[i, j, k + 1] -
                                  F[i, j, k]) / 6.0
