#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from multipy.multigrid import poisson_multigrid
from multipy.tools.util import load_problem


DEFAULT_FILE = '../problems/problem_2D_100.npy'


def main():
    U, F = load_problem(DEFAULT_FILE)
    ims = []
    oldU = U.copy()
    fig = plt.figure()
    i = 0
    for i in range(30):
        im = plt.imshow(U, cmap='RdBu_r', interpolation='nearest')
        t = plt.annotate(i, (90, 90))
        ims.append([im, t])
        U = poisson_multigrid(F, U, 0, 2, 2, 1, True)

    ani = animation.ArtistAnimation(
        fig, ims, interval=200, blit=True)
    writer = animation.PillowWriter(fps=20)
    ani.save('../graphs/heatmap.gif', writer=writer)
    # plt.show()


if __name__ == "__main__":
    main()
