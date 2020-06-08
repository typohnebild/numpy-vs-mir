#!/bin/usr/env python3
import numpy as np
import gaussSeidel as gs


def restriction(A):
    alpha = len(A.shape)

    if alpha == 1:
        return A[1::2]
    if alpha == 2:
        return A[1::2, 1::2]
    if alpha == 3:
        return A[1::2, 1::2, 1::2]
    else:
        raise ValueError('restriction: invalid dimension')


def prolongation(e, plus):
    alpha = len(e.shape)
    w = np.zeros(np.array(e.shape) * 2 + plus)

    if alpha == 1:
        w[0] = e[0] / 2
        w[1::2] = e
        w[2:-1:2] = (e[1:] + e[:-1]) / 2
        w[-1] = e[-1] / 2
    elif alpha == 2:
        e = np.pad(e, 1)
        # w[1::2, 1::2] = e[::, ::] / 2
        # w[1::2, ::2] = (e[:-1:, ::] + e[1::, ::]) / 4
        # w[::2, 1::2] = (e[::, :-1:] + e[::, 1::]) / 4
        # w[1::2, 1::2] = (e[:-1:, :-1:] + e[1::, :-1:] + e[:-1:, 1::]  + e[1::, 1::]) / 8
        for i in range(e.shape[0] - 2):
            for j in range(e.shape[1] - 2):
                w[2 * i][2 * j] = e[i][j] / 2
                w[2 * i + 1][2 * j] = (e[i][j] + e[i + 1][j]) / 4
                w[2 * i][2 * j + 1] = (e[i][j] + e[i][j + 1]) / 4
                w[2 * i + 1][2 * j + 1] = (e[i][j] + e[i][j + 1] +
                                           e[i + 1][j] + e[i + 1][j + 1]) / 8
    elif alpha == 3:
        # TODO
        raise ValueError('prolongation: dimension not implemented')
    else:
        raise ValueError('prolongation: invalid dimension')
    return w


def residual(U):
    alpha = len(U.shape)
    x = np.zeros_like(U)
    print(x)
    if alpha == 1:
        x[0] = U[0]
        x[-1] = U[-1]
        for i in range(1, U.shape[0] - 1):
            x[i] = (2.0 * U[i] - U[i - 1] - U[i + 1]) / 2.0
    elif alpha == 2:
        x[:, 0] = U[:, 0]
        x[0, :] = U[0, :]
        for i in range(1, U.shape[0] - 1):
            for j in range(1, U.shape[1] - 1):
                x[i, j] = (4.0 * U[i,j] -
                            U[i - 1, j] - U[i + 1, j] -
                            U[i, j - 1] - U[i, j + 1]) / 4.0
    elif alpha == 3:
        x[:, :, 0] = U[:, :, 0]
        x[:, 0, :] = U[:, 0, :]
        x[0, :, :] = U[0, :, :]
        for i in range(1, U.shape[0] - 1):
            for j in range(1, U.shape[1] - 1):
                for k in range(1, U.shape[2] - 1):
                    x[i, j] = (4.0 * U[i,j] -
                                U[i - 1, j, k] - U[i + 1, j, k] -
                                U[i, j - 1, k] - U[i, j + 1, k] -
                                U[i, j, k - 1] - U[i, j, k + 1]) / 8.0
    else:
        raise ValueError('residual: invalid dimension')

    return x


def multigrid(F, U, l, v1, v2, mu):
    # TODO: abfangen, dass level zu gross wird!!!
    if l == 1:
        # solve
        return gs.GS_RB(F, U=U, max_iter=1000)
    else:
        # smoothing
        U = gs.GS_RB(F, U=U, max_iter=v1)
        # residual
        r = F - residual(U)
        # restriction
        r = restriction(r)

        # recursive call
        e = np.zeros_like(r)
        for _ in range(mu):
            e = multigrid(e, r, l - 1, v1, v2, mu)
        # prolongation
        e = prolongation(e, F.shape[0] % 2)
        # correction
        U = U + e
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