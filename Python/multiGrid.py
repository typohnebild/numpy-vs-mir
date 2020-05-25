#!/bin/usr/env python3
import numpy as np
import gaussSeidel as gs


def restriction(A, r):
    return A[1::2, 1::2], r[1::2] #TODO

def prolangation(e):
    #TODO
    w = np.zeros(e.shape[0] * 2 + 1)
    w[0] = e[0]/2
    w[1::2] = e
    w[2:-1:2] = (e[1:] + e[:-1]) / 2
    w[-1] = e[-1] / 2
    return w


def multigrid(A, x, b, l, v1, v2, mu):
    if (l==1):
        #solve
        return np.linalg.solve(A,b)
    else:
        #smoothing
        x = gs.gauss_seidel(A, b, x=x, max_iter=v1)
        #residual
        r = b - A @ x
        #restriction
        A_next, r = restriction(A, r)
        print(A_next)
        print(r)

        #recursive call
        e = np.zeros_like(r)
        for _ in range(mu):
            e = multigrid(A_next, e, r, l-1, v1, v2, mu)
        print(e)
        #prolongation
        e = prolangation(e)
        #correction
        x = x + e
        #post smoothing
        return gs.gauss_seidel(A, b, x=x, max_iter=v2)

def test():
    A = np.array([[10., -1., 2.],
                  [-1., 11., -1.],
                  [2., -1., 10.]])

    b = np.array([6., 25., -11.])

    print(multigrid(A, np.zeros_like(b), b, 1,1,1,1))
    print(multigrid(A, np.zeros_like(b), b, 2,1,1,1))