#!/bin/usr/env python3
import numpy as np


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
