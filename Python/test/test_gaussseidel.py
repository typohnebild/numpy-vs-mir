import numpy as np
import pytest
from gaussSeidel import GS_RB

import gaussSeidel as gs
import heatmap as hm


def MatrixGenerator(dim, max_value=500):
    return np.random.rand(*dim) * np.random.randint(max_value)


# TODO: put some testcases here
# --- GausSeidel TestCases ---


def test_np_gs():
    """ some test stolen from wikipedia """
    A = np.array([[10., -1., 2., 0.],
                  [-1., 11., -1., 3.],
                  [2., -1., 10., -1.],
                  [0., 3., -1., 8.]])

    b = np.array([6., 25., -11., 15.])
    eps = 1e-10
    x_opt = np.linalg.solve(A, b)
    x = gs.gauss_seidel(A, b, eps)
    assert(np.linalg.norm(x_opt - x) <= eps)


def test_2D_heatMap():
    A = hm.simulate_2D(100, 500)
    hm.draw2D(A)


def test_3D_heatMap():
    A = hm.simulate_3D(100, 500)
    hm.draw3D(A)


# --- MultiGrid TestCases ---
# TODO: put some testcases here
