import numpy as np
import gaussSeidel as gs
from operators import poisson_operator_2D
import matplotlib.pyplot as plt


def initMap(dimension):
    map = np.zeros((dimension, dimension))
    map[0,:] = 1
    map[:,0] = 1
    return map

def drawMap(map):
    plt.imshow(map, cmap='hot', interpolation='nearest')
    plt.show()


def run(dim, iter):
    # get initial Heat Map
    U = initMap(dim).flatten()
    F = np.zeros(dim*dim)
    # apply Gauss Seidel on it
    A = poisson_operator_2D(dim)
    U = gs.gauss_seidel(A, U, F, max_iter=iter)
    # draw result
    return U.reshape((dim, dim))
