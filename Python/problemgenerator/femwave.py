"""
    A example problem
    solves the finite element method (wave) in NxN grid
"""
import numpy as np

import multipy.tools.apply_poisson as ap

def f(x,y):
    return np.sin(2*np.pi * x) * np.cos(2*np.pi * y)

def u(x,y):
    return f(x,y) / (8 * np.pi**2)

def create_2D(N):
    # Generate meshes
    F = f(*np.meshgrid(np.linspace(0, 1, N), np.linspace(0,1,N)))
    # Set borders correct
    F[:, 0] /= 8 * np.pi**2
    F[0, :] /= 8 * np.pi**2
    F[:, -1] /= 8 * np.pi**2
    F[-1, :] /= 8 * np.pi**2
    U = F.copy()
    U[1:-1, 1:-1] = 0
    return U, F

def solution_2D(N):
    # Generate analytical solution
    return u(*np.meshgrid(np.linspace(0, 1, N), np.linspace(0,1,N)))