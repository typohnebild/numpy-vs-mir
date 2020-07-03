"""
    Util functions
"""
import logging
import time as time
import numpy as np
import matplotlib.pyplot as plt
import cProfile

TIME_STATS = {}


logger = logging.getLogger('time')
logger.setLevel(logging.WARNING)

# TODO: ggf. auch mal mit PERF was machen

def profiling(profunc):
    def prof_wrapper(*args, **kwargs):
        with cProfile.Profile() as pr:
            value = profunc(*args, **kwargs)
        pr.print_stats()
        return value
    return prof_wrapper


def timer(func):
    def wrapper(*args, **kwargs):
        before = time.time()
        value = func(*args, **kwargs)
        after = time.time() - before
        if func.__name__ not in TIME_STATS:
            TIME_STATS[func.__name__] = [0, 0]
        TIME_STATS[func.__name__][0] += 1
        TIME_STATS[func.__name__][1] += after

        logger.info(f"{func.__name__} took {after:.6} s")

        return value
    return wrapper


def MatrixGenerator(dim, max_value=500):
    return np.random.rand(*dim) * np.random.randint(max_value)


def draw2D(U):
    if len(U.shape) == 1:
        n = int(np.sqrt(U.shape[0]))
        assert n * n == U.shape[0]
        plt.imshow(U.reshape((n, n)), cmap='RdBu_r', interpolation='nearest')
    else:
        plt.imshow(U, cmap='RdBu_r', interpolation='nearest')
    plt.show()


def draw3D(map):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface.
    for index, x in np.ndenumerate(map):
        if x > 0.5:
            ax.scatter(*index, c='black', alpha=max(x - 0.5, 0))

    fig.show()
