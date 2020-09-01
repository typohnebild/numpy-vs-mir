import numpy as np
import pytest

from .. import multigrid as mg
from ..GaussSeidel.GaussSeidel_RB import GS_RB
from ..tools import heatmap as hm
from ..tools import operators as op
from ..tools import util

# --- MultiGrid TestCases ---


def test_MG_Restriction_Prolongation_Shapes_1D_even():
    A = util.MatrixGenerator((100,))
    B = mg.restriction(A)
    C = mg.prolongation(B, A.shape)

    assert A.shape == C.shape


def test_MG_Restriction_Prolongation_Shapes_1D_odd():
    A = util.MatrixGenerator((99,))
    B = mg.restriction(A)
    C = mg.prolongation(B, A.shape)

    assert A.shape == C.shape


def test_MG_Restriction_Prolongation_Shapes_2D_even():
    A = util.MatrixGenerator((100, 100))
    B = mg.restriction(A)
    C = mg.prolongation(B, A.shape)

    assert A.shape == C.shape


def test_MG_Restriction_Prolongation_Shapes_2D_odd():
    A = util.MatrixGenerator((99, 99))
    B = mg.restriction(A)
    C = mg.prolongation(B, A.shape)

    assert A.shape == C.shape


@pytest.mark.skip("Not yet implemented")
def test_MG_Restriction_Prolongation_Shapes_3D_even():
    A = util.MatrixGenerator((100, 100, 100))
    B = mg.restriction(A)
    C = mg.prolongation(B, A.shape)

    assert A.shape == C.shape


@pytest.mark.skip("Not yet implemented")
def test_MG_Restriction_Prolongation_Shapes_3D_odd():
    A = util.MatrixGenerator((99, 99, 99))
    B = mg.restriction(A)
    C = mg.prolongation(B, A.shape)

    assert A.shape == C.shape


def test_MG_restriction_1D():
    a = np.arange(10)
    correct = np.array([0, 2, 4, 6, 8, 9])
    assert np.array_equal(mg.restriction(a), correct)
    a = np.arange(11)
    correct = np.array([0, 2, 4, 6, 8, 10])
    assert np.array_equal(mg.restriction(a), correct)


def test_MG_restriction_2D():
    def f(N): return np.arange(N * N).reshape(N, N)
    N = 5
    a = f(N)
    correct = np.array([[0, 2, 4], [10, 12, 14], [20, 22, 24]])
    assert np.array_equal(mg.restriction(a), correct)
    N = 6
    a = f(N)
    correct = np.array([[0, 2, 4, 5], [12, 14, 16, 17], [
                       24, 26, 28, 29], [30, 32, 34, 35]])
    assert np.array_equal(mg.restriction(a), correct)


def test_MG_restriction_3D():
    def f(N): return np.arange(N * N * N).reshape(N, N, N)
    N = 5
    a = f(N)
    correct = np.array([[[0., 2., 4.],
                         [10., 12., 14.],
                         [20., 22., 24.]],
                        [[50., 52., 54.],
                         [60., 62., 64.],
                         [70., 72., 74.]],
                        [[100., 102., 104.],
                         [110., 112., 114.],
                         [120., 122., 124.]]])
    assert np.array_equal(mg.restriction(a), correct)
    N = 6
    a = f(N)
    correct = np.array([[[0., 2., 4., 5.],
                         [12., 14., 16., 17.],
                         [24., 26., 28., 29.],
                         [30., 32., 34., 35.]],
                        [[72., 74., 76., 77.],
                         [84., 86., 88., 89.],
                         [96., 98., 100., 101.],
                         [102., 104., 106., 107.]],
                        [[144., 146., 148., 149.],
                         [156., 158., 160., 161.],
                         [168., 170., 172., 173.],
                         [174., 176., 178., 179.]],
                        [[180., 182., 184., 185.],
                         [192., 194., 196., 197.],
                         [204., 206., 208., 209.],
                         [210., 212., 214., 215.]]])
    assert np.array_equal(mg.restriction(a), correct)


def test_MG_weighted_restriction_1D():
    a = np.array([1., 2., 3., 2., 1.])
    assert np.array_equal(mg.weighted_restriction(a), np.array([1., 2.5, 1.]))
    b = np.array([1., 2., 3., 3., 2., 1.])
    assert np.array_equal(mg.weighted_restriction(b),
                          np.array([1., 2.75, 2.0, 1.]))
    c = np.arange(10)
    assert np.array_equal(mg.weighted_restriction(c), mg.restriction(c))
    d = np.arange(11)
    assert np.array_equal(mg.weighted_restriction(d), mg.restriction(d))


def test_MultiGrid_VS_GS_RB():
    eps = 1e-3
    N = 20
    # Variables
    U = hm.initMap_2D(N)
    F = hm.heat_sources_2D(N)

    A = GS_RB(np.copy(F), np.copy(U), eps=eps, numba=False)
    B = mg.poisson_multigrid(F, U, 2, 2, 2, 2, 100, numba=False)

    assert np.allclose(A, B, atol=eps)
