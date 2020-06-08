import pytest
import numpy as np
import gaussSeidel as gs
import heatmap as hm
import multiGrid as mg
import operators as op


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
    x = gs.gauss_seidel(A, b, eps=eps)
    assert np.allclose(x, x_opt, rtol=eps)



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

def test_MG_Restriction_Prolongation_Shapes_1D_even():
    A = MatrixGenerator((100))
    B = mg.restriction(A)
    C = mg.prolongation(B, 0)

    assert A.shape == C.shape


def test_MG_Restriction_Prolongation_Shapes_1D_odd():
    A = MatrixGenerator((99))
    B = mg.restriction(A)
    C = mg.prolongation(B, 1)

    assert A.shape == C.shape


def test_MG_Restriction_Prolongation_Shapes_2D_even():
    A = MatrixGenerator((100, 100))
    B = mg.restriction(A)
    C = mg.prolongation(B, 0)

    assert A.shape == C.shape


def test_MG_Restriction_Prolongation_Shapes_2D_odd():
    A = MatrixGenerator((99, 99))
    B = mg.restriction(A)
    C = mg.prolongation(B, 1)

    assert A.shape == C.shape


def test_MG_Restriction_Prolongation_Shapes_3D_even():
    A = MatrixGenerator((100, 100, 100))
    B = mg.restriction(A)
    C = mg.prolongation(B, 0)

    assert A.shape == C.shape


def test_MG_Restriction_Prolongation_Shapes_3D_odd():
    A = MatrixGenerator((99, 99, 99))
    B = mg.restriction(A)
    C = mg.prolongation(B, 1)

    assert A.shape == C.shape


def test_residualize():
    eps = 1e-5
    # Variables
    U = hm.initMap_2D(40)

    A = op.poisson_operator_2D(U.shape[0])
    B = A @ U.flatten()
    C = mg.residualize(U)

    assert np.allclose(C, B, rtol=eps)



def test_MultiGrid_VS_GS_RB():
    eps = 1e-5
    # Variables
    U = hm.initMap_2D(40)
    F = hm.heat_sources_2D(40)
    # Gauss Seidel Red Black
    A = gs.GS_RB(-F, U, max_iter=30)

    hm.draw2D(A)


    # MultiGrid
    U = hm.initMap_2D(40)
    F = hm.heat_sources_2D(40)
    B = mg.multigrid(-F, U, 6, 15, 15, 3)

    hm.draw2D(B)

    assert np.allclose(A, B, rtol=eps)