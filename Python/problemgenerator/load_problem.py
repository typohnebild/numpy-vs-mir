import numpy as np


def load_problem(path):
    U, F = np.load(path)
    return U, F
