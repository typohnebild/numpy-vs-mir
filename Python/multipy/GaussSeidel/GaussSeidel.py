#!/bin/usr/env python3
import numpy as np


def gauss_seidel(A, F, U=None, eps=1e-10, max_iter=1_000_000):
    """Implementation of Gauss Seidl iterations
       should solve AU = F
       @param A n x m Matrix
       @param F n vector
       @return n vector
    """
    n, *_ = A.shape
    if U is None:
        U = np.zeros_like(F)

    for _ in range(max_iter):
        U_next = np.zeros_like(U)
        for i in range(n):
            left = np.dot(A[i, :i], U_next[:i])
            right = np.dot(A[i, i + 1:], U[i + 1:])
            U_next[i] = (F[i] - left - right) / (A[i, i])

        U = U_next
        if np.linalg.norm(F - A @ U) < eps:
            break

    return U
