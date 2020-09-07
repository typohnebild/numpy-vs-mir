#!/usr/bin/env python3

import sys

from multipy.multigrid import poisson_multigrid
from multipy.tools.util import str2bool, load_problem, timer


def measure(path, numba=True):
    U, F = load_problem(path)
    poisson_multigrid(F, U, 0, 2, 2, 2, 100, numba=numba)


if __name__ == "__main__":
    path = "../problems/problem_2D_100.npy"
    numba = True
    if len(sys.argv) >= 2:
        path = sys.argv[1]
        if len(sys.argv) == 3:
            numba = str2bool(str(sys.argv[2]))
        elif len(sys.argv) > 3:
            raise ValueError("to much input parameters")
    measure(path, numba)
