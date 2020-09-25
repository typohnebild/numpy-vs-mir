#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from multipy.multigrid import PoissonCycle
from multipy.tools.util import load_problem


DEFAULT_FILE = '../problems/problem_2D_100.npy'


def main():
    U, F = load_problem(DEFAULT_FILE)
    cycle = PoissonCycle(F, 2, 2, 2, 0, True)
    ims = []
    fig = plt.figure()
    for i in range(100):
        im = plt.imshow(U, cmap='RdBu_r', interpolation='nearest')
        t = plt.annotate(i, (90, 90))
        norm = cycle.norm(U)
        n = plt.annotate(norm, (10, 10))
        ims.append([im, t, n])

        if norm <= 1e-3:
            break
        U = cycle(U)

    ani = animation.ArtistAnimation(fig, ims, interval=1000, repeat=True)
    writer = animation.PillowWriter(fps=2)
    ani.save('../graphs/heatmap.gif', writer=writer)


if __name__ == "__main__":
    main()
