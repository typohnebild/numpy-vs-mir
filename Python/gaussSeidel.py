#!/bin/usr/env python3
import numpy as np
from util import timer


@timer
def gauss_seidel(A, b, x=None, eps=1e-10, max_iter=1000):
    """Implementation of Gauss Seidl iterations
       should solve Ax = b
       @param A n x m Matrix
       @param b n vector
       @return x n vector
    """
    n, *_ = A.shape
    if x is None:
        x = np.zeros_like(b)
    for _ in range(max_iter):
        x_next = np.zeros_like(x)
        for i in range(n):
            left = np.dot(A[i, :i], x_next[:i])
            right = np.dot(A[i, i + 1:], x[i + 1:])
            x_next[i] = (b[i] - left - right) / A[i, i]

        x = x_next
        if np.linalg.norm(b - A @ x) < eps:
            break

    return x


@timer
def GS_RB(F, U=None, max_iter=1000):
    """Implementation of Gauss Seidl Red Black iterations
       should solve AU = F
       A poisson equation
       @param F n vector
       @return U n vector
    """

    if U is None:
        U = np.zeros_like(F)

    if len(F.shape) == 1:
        return GS_1D_RB(F, U, max_iter)
    if len(F.shape) == 2:
        return GS_2D_RB(F, U, max_iter)
    if len(F.shape) == 3:
        return GS_3D_RB(F, U, max_iter)

    raise ValueError("Wrong Shape!!!")


def GS_1D_RB(F, U, max_iter):
    """Implementation of 2D Red Black Gauss Seidl iterations
       should solve AU = F
       A poisson equation
       @param F n vector
       @return U n vector
    """

    # initialize dimensions
    n = F.shape[0]


    def sweep(color):
        """
        Does the sweeps
        @param color 1 = red 0 for black
        """
        for i in range(1, n - 1):
            if i % 2 == color:
                U[i] = (U[i - 1] +
                        U[i + 1] -
                        F[i]) / 2.0

    # Anzahl an Gauss-Seidel-Iterationen ausfuehren
    for _ in range(max_iter):
        # rote Halbiteration
        sweep(1)
        # schwarze Halbiteration
        sweep(0)

    return U


def GS_2D_RB(F, U, max_iter):
    """Implementation of 2D Red Black Gauss Seidl iterations
       should solve AU = F
       A poisson equation
       @param F n vector
       @return U n vector
    """

    # initialize dimensions
    m, n = F.shape

    def sweep(color):
        """
        Does the sweeps
        @param color 1 = red 0 for black
        """
        for j in range(1, n - 1):
            for i in range(1, m - 1):
                if (i + j) % 2 == color:
                    U[i, j] = (U[i - 1, j] +
                               U[i + 1, j] +
                               U[i, j - 1] +
                               U[i, j + 1] -
                               F[i, j]) / 4.0

    # Anzahl an Gauss-Seidel-Iterationen ausfuehren
    for _ in range(max_iter):
        # rote Halbiteration
        sweep(1)
        # schwarze Halbiteration
        sweep(0)

    return U


def GS_3D_RB(F, U, max_iter):
    """Implementation of 3D Red Black Gauss Seidl iterations
       should solve AU = F
       A poisson equation
       @param F n vector
       @return U n vector
    """
    # initialize dimensions
    m, n, o = F.shape

    def sweep(color):
        """
        Does the sweeps
        @param color 1 = red 0 for black
        """
        for k in range(1, o - 1):
            for j in range(1, n - 1):
                for i in range(1, m - 1):
                    if (i + j + k) % 2 == color:
                        U[i, j, k] = (U[i - 1, j, k] +
                                      U[i + 1, j, k] +
                                      U[i, j - 1, k] +
                                      U[i, j + 1, k] +
                                      U[i, j, k - 1] +
                                      U[i, j, k + 1] -
                                      F[i, j, k]) / 6.0

    # Anzahl an Gauss-Seidel-Iterationen ausfuehren
    for _ in range(max_iter):
        # rote Halbiteration
        sweep(1)
        # schwarze Halbiteration
        sweep(0)

    return U

