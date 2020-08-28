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


def test_MultiGrid_VS_GS_RB():
    eps = 1e-3
    # Variables
    U = hm.initMap_2D(40)
    F = hm.heat_sources_2D(40)

    A = GS_RB(np.copy(F), np.copy(U), eps=eps, numba=False)
    B = mg.poisson_multigrid(F, U, 2, 2, 2, 2, 100, numba=False)

    assert np.allclose(A, B, atol=eps)
