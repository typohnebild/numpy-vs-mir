#!/bin/usr/env python3
import numpy as np
from util import timer


@timer
def gauss_seidel(A, b, x=None, eps=1e-8, max_iter=1000):
    """Implementation of Gauss Seidl iterations
       should solve Ax = b
       @param A n x m Matrix
       @param b n vector
       @return x n vector
    """
    n, *_ = A.shape
    if x is None:
        x = np.zeros_like(b)
    for it in range(max_iter):
        x_next = np.zeros_like(x)
        for i in range(n):
            left = np.dot(A[i, :i], x_next[:i])
            right = np.dot(A[i, i + 1:], x[i + 1:])
            x_next[i] = (b[i] - left - right) / A[i, i]

        x = x_next
        if np.linalg.norm(b - A @ x) < eps:
            break
    print(it)
    return x


@timer
def GS_RB(F, U=None, eps=1e-8, max_iter=1000):
    """Implementation of Gauss Seidl Red Black iterations
       should solve AU = F
       A poisson equation
       @param F n vector
       @return U n vector
    """
    if len(F.shape) == 2:
        return GS_2D_RB(F, U, eps, max_iter)
    if len(F.shape) == 3:
        return GS_3D_RB(F, U, eps, max_iter)

    print("Wrong Shape!!!")


def GS_2D_RB(F, U=None, eps=1e-8, max_iter=1000):
    """Implementation of 2D Red Black Gauss Seidl iterations
       should solve AU = F
       A poisson equation
       @param F n vector
       @return U n vector
    """

    if U is None:
        U = np.zeros_like(F)

    # initialize dimensions
    m = F.shape[0]
    n = F.shape[1]

    # Anzahl an Gauss-Seidel-Iterationen ausfuehren
    for _ in range(max_iter):
        # rote Halbiteration
        for j in range(1, n - 1):
            for i in range(1, m - 1):
                if (i + j) % 2 == 1:
                    U[i, j] = (U[i - 1, j] +
                                U[i + 1, j] +
                                U[i, j - 1] +
                                U[i, j + 1] +
                                F[i, j]) / 4.0

        # schwarze Halbiteration
        for j in range(1, n - 1):
            for i in range(1, m - 1):
                if (i + j) % 2 == 0:
                    U[i, j] = (U[i - 1, j] +
                                U[i + 1, j] +
                                U[i, j - 1] +
                                U[i, j + 1] +
                                F[i, j]) / 4.0

    return U


def GS_3D_RB(F, U=None, eps=1e-8, max_iter=1000):
    """Implementation of 3D Red Black Gauss Seidl iterations
       should solve AU = F
       A poisson equation
       @param F n vector
       @return U n vector
    """

    if U is None:
        U = np.zeros_like(F)

    # initialize dimensions
    m = F.shape[0]
    n = F.shape[1]
    o = F.shape[2]

    # Anzahl an Gauss-Seidel-Iterationen ausfuehren
    for _ in range(max_iter):
        # rote Halbiteration
        for k in range(1, o - 1):
            for j in range(1, n - 1):
                for i in range(1, m - 1):
                    if (i + j + k) % 2 == 1:
                        U[i, j, k] = (U[i - 1, j, k] +
                                      U[i + 1, j, k] +
                                      U[i, j - 1, k] +
                                      U[i, j + 1, k] +
                                      U[i, j, k - 1] +
                                      U[i, j, k + 1] -
                                      F[i, j, k]) / 6.0

        # schwarze Halbiteration
        for k in range(1, o - 1):
            for j in range(1, n - 1):
                for i in range(1, m - 1):
                    if (i + j + k) % 2 == 0:
                        U[i, j, k] = (U[i - 1, j, k] +
                                      U[i + 1, j, k] +
                                      U[i, j - 1, k] +
                                      U[i, j + 1, k] +
                                      U[i, j, k - 1] +
                                      U[i, j, k + 1] -
                                      F[i, j, k]) / 6.0

    return U


def test():
    """ some test stolen from wikipedia """
    A = np.array([[10., -1., 2., 0.],
                  [-1., 11., -1., 3.],
                  [2., -1., 10., -1.],
                  [0., 3., -1., 8.]])

    b = np.array([6., 25., -11., 15.])
    eps = 1e-10
    x_opt = np.linalg.solve(A, b)
    x = gauss_seidel(A, b, eps)
    assert(np.linalg.norm(x_opt - x) <= eps)
