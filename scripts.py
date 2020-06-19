import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, LinearLocator

from Python.tools.heatmap import simulate_2D, simulate_3D
import numpy as np


def draw2D(map):
    plt.imshow(map, cmap='RdBu_r', interpolation='nearest')
    plt.show()


def draw3D(map):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface.
    for index, x in np.ndenumerate(map):
        if x > 0.5:
            ax.scatter(*index, c='black', alpha=max(x - 0.5, 0))

    fig.show()


def test_2D_heatMap():
    A = simulate_2D(10, 500)
    draw2D(A)
    return A


def test_3D_heatMap():
    A = simulate_3D(10, 500)
    draw3D(A)
    return A
