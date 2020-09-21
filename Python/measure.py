#!/usr/bin/env python3

import sys
import optparse
import time
import logging


from multipy.multigrid import poisson_multigrid
from multipy.tools.util import load_problem, timer


logging.basicConfig(level=logging.INFO)


def measure(F, U, numba=True):
    poisson_multigrid(F, U, 0, 2, 2, 2, 100, numba=numba)


def main():
    start = time.perf_counter()

    default_problem = '../problems/problem_2D_100.npy'

    parser = optparse.OptionParser()
    parser.add_option(
        '-n',
        action='store_true',
        dest='numba',
        default=False,
        help='activates numba')
    parser.add_option(
        '-v',
        action='store_true',
        dest='verbose',
        default=False,
        help='makes it more verbose')
    parser.add_option(
        '-d',
        action='store',
        dest='delay',
        type=int,
        default=500,
        help='delays the start of the run by DELAY ms (default:500)')
    parser.add_option('-p', action='store', dest='path',
                      default=default_problem,
                      help='path to a problem (npy file) that is loaded')
    options, _ = parser.parse_args()

    U, F = load_problem(options.path)
    # warm up with the smaller problem so it doesnt take to long for big
    # problems
    U1, F1 = load_problem(default_problem)
    poisson_multigrid(F1, U1, 0, 1, 1, 1, 1, numba=options.numba)

    if options.verbose:
        logging.getLogger('multipy.multigrid').setLevel(level=logging.DEBUG)
    else:
        logging.getLogger('multipy.multigrid').setLevel(level=logging.INFO)

    rest = options.delay / 1000 - (time.perf_counter() - start)
    if 0 < rest:
        time.sleep(rest)

    start = time.perf_counter()
    measure(F, U, options.numba)
    logging.info(time.perf_counter() - start)


if __name__ == '__main__':
    main()
