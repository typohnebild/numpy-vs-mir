#!/usr/bin/env python3

import time
import logging

from startup import DEFAULT_PROBLEM, getopts

from multipy.GaussSeidel.GaussSeidel_RB import GS_RB
from multipy.tools.util import load_problem, timer


logging.basicConfig(level=logging.INFO)


def measure(F, U, numba=True):
    GS_RB(
        F,
        U,
        h=1,
        max_iter=10_000_000,
        eps=1e-8,
        norm_iter=1000,
        numba=numba)


def main():
    start = time.perf_counter()
    options = getopts()

    U, F = load_problem(options.path)
    # warm up with the smaller problem so it doesnt take to long for big
    # problems
    U1, F1 = load_problem(DEFAULT_PROBLEM)
    measure(F1, U1, options.numba)

    if options.verbose:
        logging.getLogger('multipy.GaussSeidel.GaussSeidel_RB').setLevel(
            level=logging.DEBUG)
    else:
        logging.getLogger('multipy.GaussSeidel.GaussSeidel_RB').setLevel(
            level=logging.INFO)

    rest = options.delay / 1000 - (time.perf_counter() - start)
    if 0 < rest:
        time.sleep(rest)

    start = time.perf_counter()
    measure(F, U, options.numba)

    logging.info(time.perf_counter() - start)


if __name__ == '__main__':
    main()
