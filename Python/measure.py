#!/usr/bin/env python3

import sys

from multipy.multigrid import poisson_multigrid
from multipy.tools.heatmap import create_problem_2D
from multipy.tools.util import FLOPS, str2bool


def measure(N, numba=True):
    U, F = create_problem_2D(N)
    poisson_multigrid(F, U, 0, 1, 1, 1, 10, numba=numba)
    print(FLOPS)


if __name__ == "__main__":
    N = 128
    numba = False
    if len(sys.argv) >= 2:
        N = int(sys.argv[1])
        if len(sys.argv) == 3:
            numba = str2bool(str(sys.argv[2]))
        elif len(sys.argv) > 3:
            raise ValueError("to much input parameters")
    measure(N, numba)
