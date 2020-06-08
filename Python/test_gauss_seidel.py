import pytest
import numpy as np
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
    x = gs.gauss_seidel(A, b, eps=eps)
    assert np.allclose(x, x_opt, rtol=eps)


def test_2D_heatMap():
    A = hm.simulate_2D(10, 500)
    # hm.draw2D(A)
    return A


def test_3D_heatMap():
    A = hm.simulate_3D(10, 500)
    # hm.draw3D(A)
    return A


def test_red_black_one_iter():
    U = np.ones((3, 3))
    F = np.zeros((3, 3))
    F[1, 1] = 1
    expected = np.ones((3, 3))
    expected[1, 1] = 0.75
    actual = gs.GS_RB(F, U, max_iter=1)
    assert np.allclose(expected, actual, rtol=1e-8)


def test_red_black_against_gauss_seidel():
    eps = 1e-12
    N = 20
    max_iter = 1000
    A = hm.poisson_operator_2D(N)
    U = hm.initMap_2D(N)
    F = hm.heat_sources_2D(N)
    U1 = gs.gauss_seidel(A,
                         F.flatten(),
                         U.copy().flatten(),
                         eps=eps,
                         max_iter=max_iter).reshape((N, N))
    U2 = gs.GS_RB(-F, U.copy(), max_iter=max_iter)
    assert np.allclose(U1, U2, rtol=eps)


# --- MultiGrid TestCases ---
# TODO: put some testcases here