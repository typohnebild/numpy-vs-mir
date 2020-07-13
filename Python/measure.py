#!/usr/bin/python3

# import logging
from multipy.tools.heatmap import create_problem_2D
from multipy.multigrid import poisson_multigrid
import sys

# logging.basicConfig(level=logging.INFO)


def measure(N):
    U, F = create_problem_2D(N)
    poisson_multigrid(F, U, 5, 2, 2, 2, 100)


if __name__ == "__main__":
    N = 128
    if len(sys.argv) == 2:
        N = int(sys.argv[1])
    measure(N)
