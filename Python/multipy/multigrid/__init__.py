import logging

import numpy as np

from .cycle import GeneralCycle, PoissonCycle
from .prolongation import prolongation
from .restriction import restriction, weighted_restriction

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def poisson_multigrid(F, U, l, v1, v2, mu, iter_cycle, numba=True):
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
    cycle = PoissonCycle(F, v1, v2, mu, l, numba)
    eps = 1e-3
    return multigrid(cycle, U, eps, iter_cycle)


def multigrid(cycle, U, eps, iter_cycle):
    for i in range(1, iter_cycle + 1):
        U = cycle(U)
        norm = cycle.norm(U)
        logger.debug(f"Residual has a L2-Norm of {norm:.4} after {i} MGcycle")
        if norm <= eps:
            logger.info(
                f"converged after {i} cycles with {norm:.4} error")
            break
    return U


def general_multigrid(A, F, U, l, v1, v2, mu, iter_cycle):
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

    cycle = GeneralCycle(A, F, v1, v2, mu, l)
    eps = 1e-3
    return multigrid(cycle, U, eps, iter_cycle)
