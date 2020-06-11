import numpy as np
import pytest

from ..GaussSeidel.GaussSeidel import gauss_seidel
from ..GaussSeidel.GaussSeidel_RB import GS_RB
from .. import heatmap as hm
from .. import operators as op


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
    x = gauss_seidel(A, b, eps=eps)
    assert np.allclose(x, x_opt, rtol=eps)


def test_red_black_one_iter():
    U = np.ones((3, 3))
    F = np.zeros((3, 3))
    F[1, 1] = 1
    expected = np.ones((3, 3))
    expected[1, 1] = 0.75
    actual = GS_RB(F, U, max_iter=1)
    assert np.allclose(expected, actual, rtol=1e-8)


def test_red_black_against_gauss_seidel():
    eps = 1e-12
    N = 20
    max_iter = 1000
    A = hm.poisson_operator_2D(N)
    U = hm.initMap_2D(N)
    F = hm.heat_sources_2D(N)
    U1 = gauss_seidel(A,
                      F.flatten(),
                      U.copy().flatten(),
                      eps=eps,
                      max_iter=max_iter).reshape((N, N))
    U2 = GS_RB(-F, U.copy(), max_iter=max_iter)
    assert np.allclose(U1, U2, rtol=eps)
