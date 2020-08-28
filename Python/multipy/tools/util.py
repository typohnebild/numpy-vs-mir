"""
    Util functions
"""
import logging
import time as time
from functools import wraps
import numpy as np

TIME_STATS = {}
FLOPS = {}

logger = logging.getLogger('time')
logger.setLevel(logging.INFO)

# TODO: ggf. auch mal mit PERF was machen


def profiling(profunc):

    import cProfile
    from pstats import SortKey, Stats

    @wraps(profunc)
    def prof_wrapper(*args, **kwargs):
        with cProfile.Profile() as pr:
            value = profunc(*args, **kwargs)
        p = Stats(pr)
        p.sort_stats(SortKey.TIME).dump_stats(
            f"profiles/{profunc.__name__}_{args[0]}.prof")
        return value
    return prof_wrapper


def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        before = time.perf_counter()
        value = func(*args, **kwargs)
        after = time.perf_counter() - before
        if func.__name__ not in TIME_STATS:
            TIME_STATS[func.__name__] = [0, 0]
        TIME_STATS[func.__name__][0] += 1
        TIME_STATS[func.__name__][1] += after

        logger.info(f"{func.__name__}({args}) took {after:.6}")

        return value
    return wrapper


def counter(func):
    """
        A Decorator function to count the flops
        the values are just ridicusly guessed
    """
    @wraps(func)
    def counter_wrapper(*args, **kwargs):
        value = func(*args, **kwargs)
        if func.__name__ not in FLOPS:
            FLOPS[func.__name__] = 0

        # TODO: sweep1D and sweep_3D
        if (func.__name__ == "sweep_2D"):
            N = args[1].shape[0]
            FLOPS[func.__name__] += 6 * (((N - 2) // 2)**2)
        elif (func.__name__ == "weighted_restriction"):
            alpha = len(args[0].shape)
            N = args[0].shape[0] // 2 + 1
            if (alpha == 1):
                pass
            elif (alpha == 2):
                FLOPS[func.__name__] += 11 * ((N - 2) // 2)**2
            elif (alpha == 3):
                pass
        elif (func.__name__ == "prolongation"):
            alpha = len(args[1])
            N = args[1][0]
            if (alpha == 1):
                pass
            elif (alpha == 2):
                FLOPS[func.__name__] += 6 * ((N - 2) // 2)**2
            elif (alpha == 3):
                pass

        return value
    return counter_wrapper


def MatrixGenerator(dim, max_value=500):
    return np.random.rand(*dim) * np.random.randint(max_value)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

