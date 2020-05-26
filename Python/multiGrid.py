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


def prolongation(e):
    alpha = len(e.shape)
    w = np.zeros(np.array(e.shape) * 2 + 1)

    if alpha == 1:
        w[0] = e[0]/2
        w[1::2] = e
        w[2:-1:2] = (e[1:] + e[:-1]) / 2
        w[-1] = e[-1] / 2
    elif alpha == 2:
        e = np.pad(e,1)
        # w[1::2, 1::2] = e[::, ::] / 2
        # w[1::2, ::2] = (e[:-1:, ::] + e[1::, ::]) / 4
        # w[::2, 1::2] = (e[::, :-1:] + e[::, 1::]) / 4
        # w[1::2, 1::2] = (e[:-1:, :-1:] + e[1::, :-1:] + e[:-1:, 1::]  + e[1::, 1::]) / 8
        for i in range(e.shape[0] - 2):
            for j in range(e.shape[1] - 2):
                w[2*i][2*j] = e[i][j] / 2
                w[2*i + 1][2*j] = (e[i][j] + e[i+1][j]) / 4
                w[2*i][2*j + 1] = (e[i][j] + e[i][j+1]) / 4
                w[2*i + 1][2*j + 1] = (e[i][j] + e[i][j+1] +
                    e[i+1][j] + e[i+1][j+1]) / 8
    else:
        raise ValueError('prolongation: invalid dimension')
    return w


def multigrid(A, x, b, l, v1, v2, mu):
    if (l==1):
        # solve
        return np.linalg.solve(A,b)
    else:
        # smoothing
        x = gs.gauss_seidel(A, b, x=x, max_iter=v1)
        # residual
        r = b - A @ x
        # restriction
        A_next = restriction(A)
        r = restriction(r)
        print(A_next)
        print(r)

        # recursive call
        e = np.zeros_like(r)
        for _ in range(mu):
            e = multigrid(A_next, e, r, l-1, v1, v2, mu)
        print(e)
        # prolongation
        e = prolongation(e)
        # correction
        x = x + e
        # post smoothing
        return gs.gauss_seidel(A, b, x=x, max_iter=v2)


def test():
    A = np.array([[10., -1., 2.],
                  [-1., 11., -1.],
                  [2., -1., 10.]])

    b = np.array([6., 25., -11.])

    print(multigrid(A, np.zeros_like(b), b, 1,1,1,1))
    print(multigrid(A, np.zeros_like(b), b, 2,1,1,1))


def test2():
    A = np.array([[[10., -1., 2.],
                  [-1., 11., -1.],
                  [2., -1., 10.]],
                  [[10., -1., 2.],
                  [-1., 11., -1.],
                  [2., -1., 10.]]])

    b = np.array([[6., 25., -11.], [6., 25., -11.]])

    print(multigrid(A, np.zeros_like(b), b, 1,1,1,1))
    print(multigrid(A, np.zeros_like(b), b, 2,1,1,1))

