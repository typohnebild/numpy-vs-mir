import numpy as np
import gaussSeidel as gs
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
    map = initMap(dim)
    # apply Gauss Seidel on it
    # map = gs.gauss_seidel(map, max_iter=iter)
    # draw result
    drawMap(map)
