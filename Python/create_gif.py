#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from multipy.multigrid import PoissonCycle
from multipy.tools.util import load_problem


DEFAULT_FILE = '../problems/problem_2D_1024.npy'


def main():
    U, F = load_problem(DEFAULT_FILE)
    N = U.shape[0]
    cycle = PoissonCycle(F, 1, 0, 1, 0)
    ims = []
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    x = np.linspace(0, 1, 1024)
    X, Y = np.meshgrid(x, x)
    for i in range(100):
        #im = plt.imshow(U, cmap='RdBu_r', interpolation='nearest')
        im = ax.plot_surface(
            X[1: -1, 1: -1],
            Y[1: -1, 1: -1],
            U[1: -1, 1: -1],
            alpha=0.7, cmap='magma')
        t = ax.annotate(i, (0.1, 0.1), xycoords='figure fraction')
        norm = cycle.norm(U)
        n = ax.annotate(norm, (0.7, 0.1), xycoords='figure fraction')
        ims.append([im, t, n])

        if norm <= 1e-6 * N * N:
            break
        U = cycle(U)

    fig.colorbar(ims[0][0], shrink=0.5, aspect=10, pad=0.1)
    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat=True)
    writer = animation.PillowWriter(fps=2)
    ani.save('../graphs/wave.gif', writer=writer)


if __name__ == "__main__":
    main()
