import logging

import numpy as np

from ..GaussSeidel.GaussSeidel import gauss_seidel
from ..GaussSeidel.GaussSeidel_RB import GS_RB
from ..tools.apply_poisson import apply_poisson
from ..tools.operators import poisson_operator_like
from .cycle import PoissonCycle, GeneralCycle
from .prolongation import prolongation
from .restriction import restriction

logger = logging.getLogger('MG')
logger.setLevel(logging.DEBUG)


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
    cycle = PoissonCycle(F, v1, v2, mu, l)
    eps = 1e-3
    return multigrid(cycle, U, l, eps, iter_cycle)


def multigrid(cycle, U, l, eps, iter_cycle):
    for i in range(1, iter_cycle + 1):
        U = cycle(U, l)
        norm = cycle.norm(U)
        logger.debug(f"Residual has a L2-Norm of {norm:.4} after {i} MGcycle")
        if norm <= eps:
            logger.info(
                f"MG converged after {i} iterations with {norm:.4} error")
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
    return multigrid(cycle, U, l, eps, iter_cycle)