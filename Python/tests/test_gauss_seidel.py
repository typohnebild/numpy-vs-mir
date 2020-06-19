import numpy as np
import pytest

from ..GaussSeidel.GaussSeidel import gauss_seidel
from ..GaussSeidel.GaussSeidel_RB import GS_RB, sweep_1D, sweep_2D, sweep_3D
from ..tools import heatmap as hm
from ..tools import operators as op
from ..tools import util


def MatrixGenerator(dim, max_value=500):
    return np.random.rand(*dim) * np.random.randint(max_value)

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
    actual = GS_RB(F, U, h=1, max_iter=1)
    assert np.allclose(expected, actual, rtol=1e-8)


def test_red_black_against_gauss_seidel():
    eps = 1e-12
    N = 20
    max_iter = 1000

    h = 1/N

    A = hm.poisson_operator_2D(N, h)
    U = hm.initMap_2D(N)
    F = hm.heat_sources_2D(N)

    U1 = gauss_seidel(A,
                      F.flatten(),
                      U.copy().flatten(),
                      eps=eps,
                      max_iter=max_iter).reshape((N, N))
    U2 = GS_RB(-F, U.copy(), h=h, eps=eps, max_iter=max_iter)
    # TODO Warum ist das - hier wichtig???

    assert np.allclose(U1, U2, rtol=eps)


def test_sweep_1D_red():
    F = np.random.rand(10)
    U1 = np.random.rand(10)
    U2 = U1.copy()
    h = 1/10
    color = 1
    n = F.shape[0]
    for i in range(1, n - 1):
        if i % 2 == color:
            U1[i] = (U1[i - 1] +
                     U1[i + 1] -
                     F[i] * h * h) / (2.0)

    sweep_1D(color, F, U2, h)

    assert np.allclose(U1, U2)


def test_sweep_1D_black():
    F = np.random.rand(10)
    U1 = np.random.rand(10)
    U2 = U1.copy()
    h = 1/10
    color = 0
    n = F.shape[0]
    for i in range(1, n - 1):
        if i % 2 == color:
            U1[i] = (U1[i - 1] +
                     U1[i + 1] -
                     F[i] * h * h) / (2.0)

    sweep_1D(color, F, U2, h)

    assert np.allclose(U1, U2)


def test_sweep_2D_red():
    F = util.MatrixGenerator((10, 10))
    U1 = util.MatrixGenerator((10, 10))
    U2 = U1.copy()
    h = 1/100
    color = 1
    m, n = F.shape
    for j in range(1, n - 1):
        for i in range(1, m - 1):
            if (i + j) % 2 == color:
                U1[i, j] = (U1[i - 1, j] +
                            U1[i + 1, j] +
                            U1[i, j - 1] +
                            U1[i, j + 1] -
                            F[i, j] * h * h) / (4.0)
    sweep_2D(color, F, U2, h)

    assert np.allclose(U1, U2)


def test_sweep_2D_black():
    F = util.MatrixGenerator((10, 10))
    U1 = util.MatrixGenerator((10, 10))
    U2 = U1.copy()
    h = 1/100
    color = 0
    m, n = F.shape
    for j in range(1, n - 1):
        for i in range(1, m - 1):
            if (i + j) % 2 == color:
                U1[i, j] = (U1[i - 1, j] +
                            U1[i + 1, j] +
                            U1[i, j - 1] +
                            U1[i, j + 1] -
                            F[i, j] * h * h) / (4.0)
    sweep_2D(color, F, U2, h)

    assert np.allclose(U1, U2)
