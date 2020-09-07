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


def test_MG_weighted_restriction_2D():
    def f(N): return np.arange(N * N).reshape(N, N)

    c = f(5)
    assert np.array_equal(mg.weighted_restriction(c), mg.restriction(c))
    d = f(6)
    assert np.array_equal(mg.weighted_restriction(d), mg.restriction(d))


def test_MG_weighted_restriction_3D():
    def f(N): return np.arange(N * N * N).reshape(N, N, N)
    a = np.array([1.0, 2.0, 3.0, 2.0, 1.0,
                  2.0, 3.0, 4.0, 3.0, 2.0,
                  3.0, 4.0, 5.0, 4.0, 3.0,
                  4.0, 5.0, 6.0, 5.0, 4.0,
                  5.0, 6.0, 7.0, 6.0, 5.0]).reshape(5, 5)
    ret1 = mg.weighted_restriction(a)
    correct1 = np.array([[1.0, 3.0, 1.0],
                         [3.0, 4.5, 3.0],
                         [5.0, 7.0, 5.0]])
    assert np.array_equal(ret1, correct1)

    a2 = np.array([1.0, 2.0, 3.0, 3.0, 2.0, 1.0,
                   2.0, 3.0, 4.0, 4.0, 3.0, 2.0,
                   3.0, 4.0, 5.0, 5.0, 4.0, 3.0,
                   4.0, 5.0, 6.0, 6.0, 5.0, 4.0,
                   5.0, 6.0, 7.0, 7.0, 6.0, 5.0,
                   6.0, 7.0, 8.0, 8.0, 7.0, 6.0]).reshape(6, 6)
    ret2 = mg.weighted_restriction(a2)
    correct2 = np.array([[1.0, 3.0, 2.0, 1.0],
                         [3.0, 4.75, 4.0, 3.0],
                         [5.0, 6.75, 6.0, 5.0],
                         [6.0, 8.0, 7.0, 6.0]])
    assert np.array_equal(ret2, correct2)

    c = f(5)
    assert np.array_equal(mg.weighted_restriction(c), mg.restriction(c))
    d = f(6)
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


def test_MG_weighted_restriction_2D_random7():
    arr = np.array([[0.40088473, 0.89582552, 0.16398608, 0.45921818, 0.50720246,
                     0.05841615, 0.71127485],
                    [0.29420544, 0.55262016, 0.45112843, 0.63388048, 0.27870701,
                     0.43475406, 0.66402547],
                    [0.9038084, 0.16260612, 0.61827658, 0.17583573, 0.26752605,
                     0.54132342, 0.95954425],
                    [0.21255126, 0.63423338, 0.48119557, 0.42348304, 0.66583851,
                     0.80677271, 0.76529026],
                    [0.30096364, 0.36264674, 0.23783031, 0.21284939, 0.12336692,
                     0.74549574, 0.47731472],
                    [0.47082754, 0.36148885, 0.24760767, 0.968772, 0.41319792,
                     0.44027865, 0.65545125],
                    [0.98000574, 0.78919914, 0.49313159, 0.85537712, 0.44181032,
                     0.4994793, 0.97419118]])
    correct = np.array([[0.40088473, 0.16398608, 0.50720246, 0.71127485],
                        [0.9038084, 0.45367844, 0.41827524, 0.95954425],
                        [0.30096364, 0.37174358, 0.45047108, 0.47731472],
                        [0.98000574, 0.49313159, 0.44181032, 0.97419118]])
    ret = mg.weighted_restriction(arr)

    assert np.allclose(ret, correct, atol=1e-8)


def test_MG_weighted_restriction_2D_random8():
    arr = np.array([[0.81221201, 0.78276113, 0.48331298, 0.37342158, 0.69540543,
                     0.76324145, 0.82182523, 0.72875685],
                    [0.57634476, 0.31967787, 0.82186108, 0.52491243, 0.15475378,
                     0.13005756, 0.54944053, 0.2843028],
                    [0.60829286, 0.66684961, 0.03881298, 0.36623578, 0.43896866,
                     0.09926548, 0.21621183, 0.14579873],
                    [0.53163999, 0.30784403, 0.79728148, 0.5986419, 0.45822312,
                     0.61653698, 0.12602686, 0.84576779],
                    [0.99019009, 0.15173809, 0.97024363, 0.21683838, 0.32338431,
                     0.92911924, 0.76354069, 0.14346233],
                    [0.90767941, 0.2732503, 0.23990377, 0.82870636, 0.66895977,
                     0.55954603, 0.91480887, 0.56811022],
                    [0.24181791, 0.11676617, 0.65585234, 0.74380539, 0.16570513,
                     0.31648328, 0.26040337, 0.51607495],
                    [0.68527047, 0.42570696, 0.13484427, 0.48563044, 0.23751941,
                     0.60301295, 0.20323849, 0.59139025]])
    correct = np.array([[0.81221201, 0.48331298, 0.69540543, 0.82182523, 0.72875685],
                        [0.60829286, 0.450674, 0.36143624, 0.28641098, 0.14579873],
                        [0.99019009, 0.54380879, 0.5277031, 0.6169349, 0.14346233],
                        [0.24181791, 0.44420891, 0.44207825, 0.45405526, 0.51607495],
                        [0.68527047, 0.13484427, 0.23751941, 0.20323849, 0.59139025]])
    ret = mg.weighted_restriction(arr)

    assert np.allclose(ret, correct, atol=1e-8)


def test_MG_prolongation():
    A = np.random.uniform(0, 1, (4, 4))
    ret6 = mg.prolongation(A, (6, 6))
    ret7 = mg.prolongation(A, (7, 7))

    # print(A[-2::,:-1:])
    # print(ret6[-2::, ::2])

    assert np.array_equal(ret6[:1, ::2], A[:1, :-1:])
    assert np.array_equal(ret6[::2, :1], A[:-1:, :1])
    assert np.array_equal(ret6[-2::, ::2], A[-2::, :-1:])
    assert np.array_equal(ret6[::2, -2::], A[:-1:, -2::])
    assert np.array_equal(ret6[-2::, -2::], A[-2::, -2::])

    assert np.array_equal(ret7[:1, ::2], A[:1, ::])
    assert np.array_equal(ret7[::2, :1], A[::, :1])
    assert np.array_equal(ret7[-1::, ::2], A[-1::, ::])
    assert np.array_equal(ret7[::2, -1::], A[::, -1::])
