import numpy as np
from ..tools import heatmap as hm
from ..tools import operators as op
from ..tools.apply_poisson import apply_poisson


def test_apply_poisson():
    eps = 1e-12
    # Variables
    U = hm.initMap_2D(40)

    A = op.poisson_operator_2D(U.shape[0] - 2)
    B = (A @ U[1:-1, 1:-1].flatten() - hm.boundary_condition(U)
         ).reshape(np.array(U.shape) - 2)
    C = apply_poisson(-U, 1)

    assert np.allclose(C[1:-1, 1:-1], B, atol=eps)


def test_apply_poisson_1D():
    N = 50
    U = hm.initMap_1D(N)
    h = 1
    expected = np.zeros_like(U)
    expected[0] = U[0]
    expected[-1] = U[-1]
    for i in range(1, U.shape[0] - 1):
        expected[i] = (-2.0 * U[i] + U[i - 1] + U[i + 1]) / (h * h)

    assert np.array_equal(expected, apply_poisson(U, h))


def test_apply_poisson_2D():
    N = 50
    U = hm.initMap_2D(N)
    h = 1
    expected = np.zeros_like(U)
    expected[:, 0] = U[:, 0]
    expected[0, :] = U[0, :]
    for i in range(1, U.shape[0] - 1):
        for j in range(1, U.shape[1] - 1):

            expected[i, j] = (-4.0 *
                              U[i, j] +
                              U[i - 1, j] +
                              U[i + 1, j] +
                              U[i, j - 1] +
                              U[i, j + 1]) / (h * h)

    assert np.array_equal(expected, apply_poisson(U, h))


def test_apply_poisson_3D():
    N = 50
    U = hm.initMap_3D(N)
    h = 1
    expected = np.zeros_like(U)
    expected[:, :, 0] = U[:, :, 0]
    expected[:, 0, :] = U[:, 0, :]
    expected[0, :, :] = U[0, :, :]
    for i in range(1, U.shape[0] - 1):
        for j in range(1, U.shape[1] - 1):
            for k in range(1, U.shape[2] - 1):
                expected[i, j, k] = (-6.0 * U[i, j, k] +
                                     U[i - 1, j, k] +
                                     U[i + 1, j, k] +
                                     U[i, j - 1, k] +
                                     U[i, j + 1, k] +
                                     U[i, j, k - 1] +
                                     U[i, j, k + 1]) / (h * h)
    assert np.array_equal(expected, apply_poisson(U, h))
