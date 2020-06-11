#!/bin/usr/env python3


import numpy as np
import operators as op
import gaussSeidel as gs
import operators as op


def restriction(A):
    alpha = len(A.shape)
    ret = np.zeros(np.array(A.shape) // 2 + 1)
    end = ret.shape[0] - (A.shape[0] + 1) % 2

    if alpha == 1:
        ret[:end:] = A[::2]
        ret[-1] = A[-1]
    elif alpha == 2:
        ret[:end:, :end:] = A[::2, ::2]
        ret[:end, -1] = A[::2, -1]
        ret[-1, :end] = A[-1, ::2]
        ret[-1, -1] = A[-1, -1]
    elif alpha == 3:
        ret[:end:, :end:, :end:] = A[::2, ::2, ::2]

        ret[:end, :end, -1] = A[::2, ::2, -1]
        ret[-1, :end, :end] = A[-1, ::2, ::2]
        ret[:end, -1, :end] = A[::2, -1, ::2]

        ret[:end, -1, -1] = A[::2, -1, -1]
        ret[-1, :end, -1] = A[-1, ::2, -1]
        ret[-1, -1, :end] = A[-1, -1, ::2]

        ret[-1, -1, -1] = A[-1, -1, -1]
    else:
        raise ValueError('restriction: invalid dimension')
    return ret


def weighted_restriction(A):
    alpha = len(A.shape)
    B = np.zeros(np.array(A.shape) // 2)

    if alpha == 1:
        # TODO
        return A[1::2]
    if alpha == 2:
        for i in range(B.shape[0]):
            for j in range(B.shape[1]):
                B[i][j] = (A[2 * i][2 * j] / 2 +
                           (A[2 * i + 1][2 * j] + A[2 * i - 1][2 * j] +
                            A[2 * i][2 * j + 1] + A[2 * i][2 * j - 1]
                            ) / 4 +
                           (A[2 * i + 1][2 * j + 1] + A[2 * i + 1][2 * j - 1] +
                            A[2 * i - 1][2 * j + 1] + A[2 * i - 1][2 * j - 1]
                            ) / 8)
        return B
    if alpha == 3:
        # TODO
        return A[1::2, 1::2, 1::2]
    else:
        raise ValueError('weighted restriction: invalid dimension')

# in:
def prolongation(e, fine_shape):
    alpha = len(e.shape)
    w = np.zeros(fine_shape)
    end = e.shape[0] - (w.shape[0] + 1) % 2
    wend = w.shape[0] - (w.shape[0] + 1) % 2

    if alpha == 1:
        # w[0] = e[0] / 2
        w[:-1:2] = e[:-1]
        w[1:-1:2] = (e[:end - 1] + e[1:end]) / 2
        w[-1] = e[-1]
    elif alpha == 2:
        w[:-1:2, :-1:2] = e[:-1, :-1]

        w[:-1:2, -1] = e[:-1, -1]
        w[-1, :-1:2] = e[-1, :-1]
        w[-1, -1] = e[-1, -1]

        # horizontal
        w[:-1:2, 1:-1:2] = (e[:-1, :end - 1] + e[:-1, 1:end]) / 2
        w[-1, 1:-1:2] = (e[-1, :end - 1] + e[-1, 1:end]) / 2

        # vertical
        w[1:-1:2, :-1:2] = (e[:end - 1, :-1] + e[1:end, :-1]) / 2
        w[1:-1:2, -1] = (e[:end - 1, -1] + e[1:end, -1]) / 2

        w[1:-1:2, 1:-1:2] = (w[2:wend:2, 1:wend:2] +
                             w[:wend - 1:2, 1:wend:2] +
                             w[1:wend:2, :wend - 1:2] +
                             w[1:wend:2, 2:wend:2]) / 4
    elif alpha == 3:
        # TODO
        raise ValueError('prolongation: dimension not implemented')
    else:
        raise ValueError('prolongation: invalid dimension')
    return w/4


def residualize(U):
    alpha = len(U.shape)
    x = np.zeros_like(U)
    # print(x)
    if alpha == 1:
        x[0] = U[0]
        x[-1] = U[-1]
        for i in range(1, U.shape[0] - 1):
            x[i] = (2.0 * U[i] - U[i - 1] - U[i + 1])
    elif alpha == 2:
        x[:, 0] = U[:, 0]
        x[0, :] = U[0, :]
        for i in range(1, U.shape[0] - 1):
            for j in range(1, U.shape[1] - 1):
                x[i, j] = (4.0 * U[i, j] -
                           U[i - 1, j] - U[i + 1, j] -
                           U[i, j - 1] - U[i, j + 1])
    elif alpha == 3:
        x[:, :, 0] = U[:, :, 0]
        x[:, 0, :] = U[:, 0, :]
        x[0, :, :] = U[0, :, :]
        for i in range(1, U.shape[0] - 1):
            for j in range(1, U.shape[1] - 1):
                for k in range(1, U.shape[2] - 1):
                    x[i, j] = (4.0 * U[i, j] -
                               U[i - 1, j, k] - U[i + 1, j, k] -
                               U[i, j - 1, k] - U[i, j + 1, k] -
                               U[i, j, k - 1] - U[i, j, k + 1])
    else:
        raise ValueError('residual: invalid dimension')

    return x


def multigrid(F, U, l, v1, v2, mu):
    """Implementation of MultiGrid iterations
       should solve AU = F
       A is poisson equation
       @param U n x n Matrix
       @param U n x n Matrix
       @param v1 Gauss Seidel iterations in pre smoothing
       @param v2 Gauss Seidel iterations in post smoothing
       @param mu iterations for recursive call
       @return x n vector
    """

    # abfangen, dass Level zu gross wird
    if l <= 1 or U.shape[0] <= 1:
        # solve
        return gs.GS_RB(F, U=U, max_iter=1000)

    # smoothing
    U = gs.GS_RB(F, U=U, max_iter=v1)
    # residual
    r = F - residualize(U)
    # restriction
    r = restriction(r)

    # recursive call
    e = np.zeros_like(r)
    for _ in range(mu):
        e = multigrid(e, np.copy(r), l - 1, v1, v2, mu)
    # prolongation
    e = prolongation(e, U.shape)

    # do not update border
    e[0, :] = e[:, 0] = e[-1, :] = e[:, -1] = 0

    # correction
    U = U - e
    # post smoothing
    return gs.GS_RB(F, U=U, max_iter=v2)


# --- Some minor Tests ---
def test_1D():
    """
    A = np.array([[10., -1., 2.],
                  [-1., 11., -1.],
                  [2., -1., 10.]])
    """

    b = np.array([6., 25., -11.])

    print(multigrid(np.zeros_like(b), b, 1, 1, 1, 1))
    print(multigrid(np.zeros_like(b), b, 2, 1, 1, 1))


def test_2D():
    """
    A = np.array([[[10., -1., 2.],
                   [-1., 11., -1.],
                   [2., -1., 10.]],
                  [[10., -1., 2.],
                   [-1., 11., -1.],
                   [2., -1., 10.]]])
    """
    b = np.array([[6., 25., -11.], [6., 25., -11.], [6, 25, -11]])

    print(multigrid(np.zeros_like(b), b, 1, 1, 1, 1))
    print(multigrid(np.zeros_like(b), b, 2, 1, 1, 1))


def f(n):
    return np.arange(n * n).reshape((n, n))


if __name__ == "__main__":
    # A = f(7)
    A = np.ones((7,7))
    B = restriction(A)
    print(B)
    C = prolongation(B, A.shape)
    print(C)
