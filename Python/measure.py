#!/usr/bin/env python3

import sys
import optparse
import time

from multipy.multigrid import poisson_multigrid
from multipy.tools.util import load_problem, timer


def measure(F, U, numba=True):
    poisson_multigrid(F, U, 0, 2, 2, 2, 100, numba=numba)


def main():
    start = time.perf_counter()
    parser = optparse.OptionParser()
    parser.add_option('-n', action='store_true', dest='numba', default=False)
    parser.add_option(
        '-d',
        action='store',
        dest='delay',
        type=int,
        default=500)
    parser.add_option('-p', action='store', dest='path',
                      default='../problems/problem_1D_100.npy')
    options, _ = parser.parse_args()

    U, F = load_problem(options.path)
    # warm up
    poisson_multigrid(F, U.copy(), 0, 2, 2, 2, 1, numba=options.numba)

    rest = options.delay / 1000 - (time.perf_counter() - start)
    if 0 < rest:
        time.sleep(rest)

    print(time.perf_counter() - start)
    measure(F, U, options.numba)
    print(time.perf_counter() - start)


if __name__ == '__main__':
    main()
