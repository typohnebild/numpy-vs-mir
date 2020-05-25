#!/bin/usr/env python3
import numpy as np


def gauss_seidel(A, b, eps=1e-8, max_iter=1000):
    """Implementation of Gauss Seidl iterations
       should solve Ax = b
       @param A n x m Matrix
       @param b n vector
       @return x n vector
    """
    n, _ = A.shape
    x = np.zeros_like(b)
    for _ in range(max_iter):
        x_next = np.zeros_like(x)
        for i in range(n):
            left = np.dot(A[i, :i], x_next[:i])
            right = np.dot(A[i, i+1:], x[i+1:])
            x_next[i] = (b[i] - left - right)/A[i, i]

        x = x_next
        if np.linalg.norm(b - A @ x) < eps:
            break
    return x


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
