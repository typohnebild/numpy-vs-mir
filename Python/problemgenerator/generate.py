#!/usr/bin/env python3

import sys
import os
import numpy as np

from heatmap import create_problem_1D, create_problem_2D, create_problem_3D
from femwave import create_2D

def save_to_npy(file, tensor):
    np.save(file, tensor)


def generate_1D_problem(N):
    U, F = create_problem_1D(N)
    return np.array([U, F])


def generate_2D_problem(N):
    U, F = create_problem_2D(N)
    return np.array([U, F])


def generate_3D_problem(N):
    U, F = create_problem_3D(N)
    return np.array([U, F])


def generate_problem(dim):
    if dim == 1:
        return generate_1D_problem
    if dim == 2:
        return generate_2D_problem
    if dim == 3:
        return generate_3D_problem

    raise ValueError(f"{dim} is invalid dimension")


def save_problem(base, dim, tensor):
    filename = f"{base}/problem_{dim}D_{N}"
    if os.path.exists(filename):
        os.remove(filename)
    save_to_npy(filename, tensor)


if __name__ == "__main__":
    if not len(sys.argv) == 4:
        print(f"{sys.argv[0]} base dimension N")
        exit(1)
    base, dim, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    save_problem(base, dim, generate_problem(dim)(N))
