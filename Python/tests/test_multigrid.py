import numpy as np
import pytest

from ..import multigrid as mg
from ..GaussSeidel.GaussSeidel_RB import GS_RB
from ..import heatmap as hm
from ..import operators as op


def MatrixGenerator(dim, max_value=500):
    return np.random.rand(*dim) * np.random.randint(max_value)

# --- MultiGrid TestCases ---


def test_MG_Restriction_Prolongation_Shapes_1D_even():
    A = MatrixGenerator((100,))
    B = mg.restriction(A)
    C = mg.prolongation(B, A.shape)

    assert A.shape == C.shape


def test_MG_Restriction_Prolongation_Shapes_1D_odd():
    A = MatrixGenerator((99,))
    B = mg.restriction(A)
    C = mg.prolongation(B, A.shape)

    assert A.shape == C.shape


def test_MG_Restriction_Prolongation_Shapes_2D_even():
    A = MatrixGenerator((100, 100))
    B = mg.restriction(A)
    C = mg.prolongation(B, A.shape)

    assert A.shape == C.shape


def test_MG_Restriction_Prolongation_Shapes_2D_odd():
    A = MatrixGenerator((99, 99))
    B = mg.restriction(A)
    C = mg.prolongation(B, A.shape)

    assert A.shape == C.shape


@pytest.mark.skip("Not yet implemented")
def test_MG_Restriction_Prolongation_Shapes_3D_even():
    A = MatrixGenerator((100, 100, 100))
    B = mg.restriction(A)
    C = mg.prolongation(B, A.shape)

    assert A.shape == C.shape


@pytest.mark.skip("Not yet implemented")
def test_MG_Restriction_Prolongation_Shapes_3D_odd():
    A = MatrixGenerator((99, 99, 99))
    B = mg.restriction(A)
    C = mg.prolongation(B, A.shape)

    assert A.shape == C.shape


def test_apply_poisson():
    eps = 1e-12
    # Variables
    U = hm.initMap_2D(40)

    A = op.poisson_operator_2D(U.shape[0])
    B = (A @ U.flatten()).reshape(U.shape)
    C = mg.apply_poisson(U)

    assert np.allclose(C, B, rtol=eps)


@pytest.mark.skip("Not yet implemented")
def test_MultiGrid_VS_GS_RB():
    eps = 1e-5
    # Variables
    U = hm.initMap_2D(40)
    F = hm.heat_sources_2D(40)
    # Gauss Seidel Red Black
    A = GS_RB(-np.copy(F), np.copy(U), max_iter=30)

    # hm.draw2D(A)

    # MultiGrid
    B = mg.multigrid(-F, U, 2, 15, 15, 1)

    # hm.draw2D(B)

    assert np.allclose(A, B, rtol=eps)
