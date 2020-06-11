#!/bin/usr/env python3


import numpy as np
from ..GaussSeidel.GaussSeidel_RB import GS_RB
from ..GaussSeidel.GaussSeidel import gauss_seidel

from .restriction import restriction
from .prolongation import prolongation
from .helper import apply_poisson


def poisson_multigrid(F, U, l, v1, v2, mu):
    """Implementation of MultiGrid iterations
       should solve AU = F
       A is poisson equation
       @param U n x n Matrix
       @param U n x n Matrix
       @param v1 Gauss Seidel iterations in pre smoothing
       @param v2 Gauss Seidel iterations in post smoothing
       @param mu iterations for recursive call
       @return x n vector
    """

    # abfangen, dass Level zu gross wird
    if l <= 1 or U.shape[0] <= 1:
        # solve
        return GS_RB(F, U=U, max_iter=1000)

    # smoothing
    U = GS_RB(F, U=U, max_iter=v1)
    # residual
    r = F - apply_poisson(U)
    # restriction
    r = restriction(r)

    # recursive call
    e = np.zeros_like(r)
    for _ in range(mu):
        e = poisson_multigrid(e, np.copy(r), l - 1, v1, v2, mu)
    # prolongation
    e = prolongation(e, U.shape)

    # do not update border
    e[0, :] = e[:, 0] = e[-1, :] = e[:, -1] = 0

    # correction
    U = U - e
    # post smoothing
    return GS_RB(F, U=U, max_iter=v2)


def general_multigrid(A, F, U, l, v1, v2, mu):
    """Implementation of MultiGrid iterations
       should solve AU = F
       A is poisson equation
       @param U n x n Matrix
       @param U n x n Matrix
       @param v1 Gauss Seidel iterations in pre smoothing
       @param v2 Gauss Seidel iterations in post smoothing
       @param mu iterations for recursive call
       @return x n vector
    """

    # abfangen, dass Level zu gross wird
    if l <= 1 or U.shape[0] <= 1:
        # solve
        return gauss_seidel(A, F, U, max_iter=1000)

    # smoothing
    U = gauss_seidel(A, F, U, max_iter=v1)
    # residual
    r = F - A @ U
    # restriction
    r = restriction(r)

    # recursive call
    e = np.zeros_like(r)
    for _ in range(mu):
        e = general_multigrid(restriction(A), e, np.copy(r), l - 1, v1, v2, mu)
    # prolongation
    e = prolongation(e, U.shape)

    # do not update border
    # e[0, :] = e[:, 0] = e[-1, :] = e[:, -1] = 0

    # correction
    U = U + e
    # post smoothing
    return gauss_seidel(A, F, U, max_iter=v2)
