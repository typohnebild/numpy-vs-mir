import numpy as np


def apply_poisson(U, h=None):
    """Apply the 2D poisson operator to U."""
    alpha = len(U.shape)
    x = np.empty_like(U)

    if h is None:
        h = 1 / U.shape[0]

    if alpha == 1:
        x[0] = U[0]
        x[-1] = U[-1]
        x[1:-1] = (-2.0 * U[1:-1] + U[:-2] + U[2:]) / (h * h)
    elif alpha == 2:
        x[:, 0] = U[:, 0]
        x[0, :] = U[0, :]
        x[:, -1] = U[:, -1]
        x[-1, :] = U[-1, :]
        x[1:-1, 1:-1] = (-4.0 *
                         U[1:-1, 1:-1] +
                         U[:-2, 1:-1] +
                         U[2:, 1:-1] +
                         U[1:-1, :-2] +
                         U[1:-1, 2:]) / (h * h)

    elif alpha == 3:
        x[:, :, 0] = U[:, :, 0]
        x[:, 0, :] = U[:, 0, :]
        x[0, :, :] = U[0, :, :]
        x[:, :, -1] = U[:, :, -1]
        x[:, -1, :] = U[:, -1, :]
        x[-1, :, :] = U[-1, :, :]
        x[1:-1, 1:-1, 1:-1] = (-6.0 * U[1:-1, 1:-1, 1:-1] +
                               U[:-2, 1:-1, 1:-1] +
                               U[2:, 1:-1, 1:-1] +
                               U[1:-1, :-2, 1:-1] +
                               U[1:-1, 2:, 1:-1] +
                               U[1:-1, 1:-1, :-2] +
                               U[1:-1, 1:-1, 2:]) / (h * h)
    else:
        raise ValueError('residual: invalid dimension')

    return x
