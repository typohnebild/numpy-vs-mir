"""
    Util functions
"""
import time as time


def timer(func):
    def wrapper(*args, **kwargs):
        before = time.time()
        value = func(*args, **kwargs)
        after = time.time() - before
        print(f"{func.__name__} took {after}")

        return value
    return wrapper
