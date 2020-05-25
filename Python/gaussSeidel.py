#!/bin/usr/env python3
import numpy as np


def gauss_seidl(A, b, eps, max_iter):
    """ Implementation of Gauss seidl iterations """
    n, _ = A.shape
    x = np.zeros_like(b)
    for _ in range(max_iter):
        x_next = np.zeros_like(x)
        for i in range(n):
            left = np.dot(A[i, :i], x_next[:i])
            right = np.dot(A[i, i+1], x[i+1:])
            x_next[i] = (b[i] - left - right)/A[i, i]

        x = x_next
        if np.linalg.norm(b - A @ x_next) < eps:
            break
    return x
