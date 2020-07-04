import logging

import numpy as np

from ..GaussSeidel.GaussSeidel import gauss_seidel
from ..GaussSeidel.GaussSeidel_RB import GS_RB
from ..tools.apply_poisson import apply_poisson
from ..tools.operators import poisson_operator_like
from .cycle import Cycle
from .prolongation import prolongation
from .restriction import restriction

logger = logging.getLogger('MG')


def poisson_multigrid(F, U, l, v1, v2, mu, iter_cycle):
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
    cycle = Cycle(v1, v2, mu)
    h = 1 / U.shape[0]
    eps = 1e-3

    for i in range(1, iter_cycle + 1):
        U = cycle(F, U, l, h)
        residual = F - apply_poisson(U, h)
        norm = np.linalg.norm(residual[1:-1, 1:-1])
        logger.info(f"Residual has a L2-Norm of {norm:.4} after {i} MGcycle")
        if norm <= eps:
            logger.info(
                f"MG converged after {i} iterations with {norm:.4} error")
            break
    return U


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
        # return gauss_seidel(A, F, U, max_iter=100000)
        return np.linalg.solve(A, F)

    # smoothing
    U = gauss_seidel(A, F, U, max_iter=v1)
    # residual
    r = F - A @ U
    # restriction
    r = restriction(r)
    # TODO What about distance between grid points
    # r = 1 / (r.shape[0]**2) * r

    # recursive call
    e = np.zeros_like(r)
    for _ in range(mu):
        e = general_multigrid(
            poisson_operator_like(r),
            np.copy(r),
            np.copy(e),
            l - 1,
            v1,
            v2,
            mu)
    # print(np.linalg.norm(e), np.linalg.norm(r))
    # prolongation
    e = prolongation(e, U.shape)

    # do not update border
    # e[0, :] = e[:, 0] = e[-1, :] = e[:, -1] = 0

    # correction
    U = U + e

    # post smoothing
    return gauss_seidel(A, F, U, max_iter=v2)
