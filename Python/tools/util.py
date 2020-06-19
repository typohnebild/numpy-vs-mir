"""
    Util functions
"""
import time as time
import numpy as np


def timer(func):
    def wrapper(*args, **kwargs):
        before = time.time()
        value = func(*args, **kwargs)
        after = time.time() - before
        print(f"{func.__name__} took {after}")

        return value
    return wrapper


def MatrixGenerator(dim, max_value=500):
    return np.random.rand(*dim) * np.random.randint(max_value)
