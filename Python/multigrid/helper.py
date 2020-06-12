import numpy as np


def apply_poisson(U, h=None):
    """ applies the 2D poisson operator to U """
    alpha = len(U.shape)
    x = np.zeros_like(U)

    if h is None:
        h = 1/U.shape[0]

    if alpha == 1:
        x[0] = U[0]
        x[-1] = U[-1]
        for i in range(1, U.shape[0] - 1):
            x[i] = (2.0 * U[i] - U[i - 1] - U[i + 1]) / (h * h)
    elif alpha == 2:
        x[:, 0] = U[:, 0]
        x[0, :] = U[0, :]
        for i in range(1, U.shape[0] - 1):
            for j in range(1, U.shape[1] - 1):
                x[i, j] = (4.0 * U[i, j] -
                           U[i - 1, j] - U[i + 1, j] -
                           U[i, j - 1] - U[i, j + 1]) / (h * h)
    elif alpha == 3:
        x[:, :, 0] = U[:, :, 0]
        x[:, 0, :] = U[:, 0, :]
        x[0, :, :] = U[0, :, :]
        for i in range(1, U.shape[0] - 1):
            for j in range(1, U.shape[1] - 1):
                for k in range(1, U.shape[2] - 1):
                    x[i, j] = (4.0 * U[i, j] -
                               U[i - 1, j, k] - U[i + 1, j, k] -
                               U[i, j - 1, k] - U[i, j + 1, k] -
                               U[i, j, k - 1] - U[i, j, k + 1]) / (h * h)
    else:
        raise ValueError('residual: invalid dimension')

    return x
