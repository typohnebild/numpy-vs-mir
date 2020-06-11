import numpy as np


def prolongation(e, fine_shape):
    """
        This interpolates/ prolongates to a grid of fine_shape
        @param e
        @param fine_shape targeted shape
        @return grid with fine_shape
    """
    alpha = len(e.shape)
    w = np.zeros(fine_shape)
    end = e.shape[0] - (w.shape[0] + 1) % 2
    wend = w.shape[0] - (w.shape[0] + 1) % 2

    if alpha == 1:
        w[:-1:2] = e[:-1]
        w[1:-1:2] = (e[:end - 1] + e[1:end]) / 2
        w[-1] = e[-1]
    elif alpha == 2:
        w[:-1:2, :-1:2] = e[:-1, :-1]

        w[:-1:2, -1] = e[:-1, -1]
        w[-1, :-1:2] = e[-1, :-1]
        w[-1, -1] = e[-1, -1]

        # horizontal
        w[:-1:2, 1:-1:2] = (e[:-1, :end - 1] + e[:-1, 1:end]) / 2
        w[-1, 1:-1:2] = (e[-1, :end - 1] + e[-1, 1:end]) / 2

        # vertical
        w[1:-1:2, :-1:2] = (e[:end - 1, :-1] + e[1:end, :-1]) / 2
        w[1:-1:2, -1] = (e[:end - 1, -1] + e[1:end, -1]) / 2

        # average of 4 neighbors
        w[1:-1:2, 1:-1:2] = (w[2:wend:2, 1:wend:2] +
                             w[:wend - 1:2, 1:wend:2] +
                             w[1:wend:2, :wend - 1:2] +
                             w[1:wend:2, 2:wend:2]) / 4
    elif alpha == 3:
        # TODO
        raise ValueError('prolongation: dimension not implemented')
    else:
        raise ValueError('prolongation: invalid dimension')
    return w
