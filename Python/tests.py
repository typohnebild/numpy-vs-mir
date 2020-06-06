import numpy as np
import pytest


def MatrixGenerator(dim, max_value=500):
    return np.random.rand(*dim) * np.random.randint(max_value)


# TODO: put some testcases here