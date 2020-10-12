#!/usr/bin/env python3

import time
import logging

from startup import DEFAULT_PROBLEM, getopts, wait

from multipy.multigrid import poisson_multigrid
from multipy.tools.util import load_problem, timer


logging.basicConfig(level=logging.INFO)


def measure(F, U, numba=True):
    poisson_multigrid(F, U, 0, 2, 2, 2, 100, numba=numba)


def main():
    options = getopts()

    U, F = load_problem(options.path)
    # warm up with the smaller problem so it doesnt take to long for big
    # problems
    U1, F1 = load_problem(DEFAULT_PROBLEM)
    poisson_multigrid(F1, U1, 0, 1, 1, 1, 1, numba=options.numba)

    if options.verbose:
        logging.getLogger('multipy.multigrid').setLevel(level=logging.DEBUG)
    else:
        logging.getLogger('multipy.multigrid').setLevel(level=logging.INFO)

    wait(options)

    start = time.perf_counter()
    measure(F, U, options.numba)

    logging.info(time.perf_counter() - start)


if __name__ == '__main__':
    main()
