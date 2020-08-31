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
    assert np.allclose(x, x_opt, atol=eps)


def test_red_black_one_iter():
    U = np.ones((3, 3))
    F = np.zeros((3, 3))
    F[1, 1] = 1
    expected = np.ones((3, 3))
    expected[1, 1] = 0.75
    actual = GS_RB(F, U, h=1, max_iter=1, numba=False)
    assert np.allclose(expected, actual, atol=1e-8)


def test_gauss_seidel_vs_linalg():
    eps = 1e-12
    N = 20
    max_iter = 1000

    # h = 1 / N

    grid = hm.initMap_2D(N)
    rhs = hm.heat_sources_2D(N)
    A, U, F = hm.reshape_grid(grid, rhs)

    U1 = gauss_seidel(A,
                      F,
                      U,
                      eps=eps,
                      max_iter=max_iter)
    U2 = np.linalg.solve(A, F)

    assert np.allclose(U1, U2, atol=1e-8)


def test_gauss_seidel_vs_F():
    eps = 1e-12
    N = 20
    max_iter = 10000

    # h = 1 / N

    grid = hm.initMap_2D(N)
    rhs = hm.heat_sources_2D(N)
    A, U, F = hm.reshape_grid(grid, rhs)

    U1 = gauss_seidel(A,
                      F,
                      U,
                      eps=eps,
                      max_iter=max_iter)
    X = A @ U1
    assert np.allclose(X, F, atol=1e-7)


def test_red_black_vs_linalg():
    eps = 1e-12
    N = 20
    max_iter = 1000

    h = 1 / N

    grid = hm.initMap_2D(N)
    rhs = hm.heat_sources_2D(N)
    A, _, F = hm.reshape_grid(grid, rhs, h)

    # Linalg
    U1 = np.linalg.solve(A, F)
    # Red Black
    U2 = GS_RB(
        -rhs,
        grid.copy(),
        h=h,
        eps=eps,
        max_iter=max_iter, numba=False)[
        1:-1,
        1:-1].flatten()

    assert np.allclose(U1, U2, atol=1e-8)


def test_red_black_vs_F():
    eps = 1e-12
    N = 20
    max_iter = 1000

    h = 1 / N

    grid = hm.initMap_2D(N)
    rhs = hm.heat_sources_2D(N)
    A, _, F = hm.reshape_grid(grid, rhs, h)

    # Red Black
    U1 = GS_RB(
        -rhs,
        grid.copy(),
        h=h,
        eps=eps,
        max_iter=max_iter, numba=False)[
        1:-1,
        1:-1].flatten()
    X = A @ U1
    Y = F
    assert np.allclose(X, Y, atol=1e-7)


def test_red_black_against_gauss_seidel():
    eps = 1e-12
    N = 20
    max_iter = 1000

    h = 1. / N

    grid = hm.initMap_2D(N)
    rhs = hm.heat_sources_2D(N)
    A, U, F = hm.reshape_grid(grid, rhs, h)

    U1 = gauss_seidel(A,
                      F,
                      U,
                      eps=eps,
                      max_iter=max_iter).reshape((N - 2, N - 2))
    U2 = GS_RB(-rhs, grid.copy(), h=h, eps=eps, max_iter=max_iter, numba=False)

    assert np.allclose(U1, U2[1:-1, 1:-1], atol=1e-8)


def test_sweep_1D_red():
    N = 10
    F = np.random.rand(N)
    U1 = np.random.rand(N)
    U2 = U1.copy()
    h = 1 / N
    color = 1
    n = F.shape[0]
    for i in range(1, n - 1):
        if i % 2 == color:
            U1[i] = (U1[i - 1] +
                     U1[i + 1] -
                     F[i] * h * h) / (2.0)

    sweep_1D.py_func(color, F, U2, h * h)

    assert np.allclose(U1, U2)


def test_sweep_1D_black():
    N = 10
    F = np.random.rand(N)
    U1 = np.random.rand(N)
    U2 = U1.copy()
    h = 1 / N
    color = 0
    n = F.shape[0]
    for i in range(1, n - 1):
        if i % 2 == color:
            U1[i] = (U1[i - 1] +
                     U1[i + 1] -
                     F[i] * h * h) / (2.0)

    sweep_1D.py_func(color, F, U2, h * h)

    assert np.allclose(U1, U2)


def test_sweep_2D_red():
    N = 10
    F = util.MatrixGenerator((N, N))
    U1 = util.MatrixGenerator((N, N))
    U2 = U1.copy()
    h = 1 / N
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
    sweep_2D.py_func(color, F, U2, h * h)

    assert np.allclose(U1, U2)


def test_sweep_2D_black():
    N = 10
    F = util.MatrixGenerator((N, N))
    U1 = util.MatrixGenerator((N, N))
    U2 = U1.copy()
    h = 1 / N
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
    sweep_2D.py_func(color, F, U2, h * h)

    assert np.allclose(U1, U2)
