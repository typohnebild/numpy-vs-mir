import numpy as np


def GS_RB(F, U=None, h=None, max_iter=1000, eps=1e-10):
    """Implementation of Gauss Seidl Red Black iterations
       should solve AU = F
       A poisson equation
       @param F n vector
       @param h is distance between grid points | default is 1/N
       @return U n vector
    """

    if U is None:
        U = np.zeros_like(F)
    if h is None:
        h = 1 / (U.shape[0])

    if len(F.shape) == 1:
        # do the sweep
        sweep = sweep_1D
    elif len(F.shape) == 2:
        # do the sweep
        sweep = sweep_2D
    elif len(F.shape) == 3:
        # Anzahl an Gauss-Seidel-Iterationen ausfuehren
        sweep = sweep_3D
    else:
        raise ValueError("Wrong Shape!!!")

    # last_U = None
    # Anzahl an Gauss-Seidel-Iterationen ausfuehren
    for _ in range(max_iter):
        # rote Halbiteration
        sweep(0, F, U, h)
        # schwarze Halbiteration
        sweep(1, F, U, h)

        # pruefe Abbruchkriterium
        # if last_U is not None and np.linalg.norm(U - last_U) < eps:
        #     return U

        # last_U = U
    return U


# --- 1D Fall ---
def sweep_1D(color, F, U, h):
    """
    Does the sweeps
    @param color 1 = red 0 for black
    @param h is distance between grid points
    """
    n = F.shape[0]
    if color == 1:
        # red
        U[1:n - 1:2] = (U[0:n - 2:2] + U[2::2] - F[1:n - 1:2] * h * h) / (2.0)
    else:
        # black
        U[2:n - 1:2] = (U[1:n - 2:2] + U[3::2] - F[2:n - 1:2] * h * h) / (2.0)
# ----------------

# --- 2D Fall ---


def sweep_2D(color, F, U, h):
    """
    Does the sweeps
    @param color 1 = red 0 for black
    @param h is distance between grid points
    """

    m, n = F.shape

    if color == 1:
        # red
        U[1:m - 1:2, 2:n - 1:2] = (U[0:m - 2:2, 2:n - 1:2] +
                                   U[2::2, 2:n - 1:2] +
                                   U[1:m - 1:2, 1:n - 2:2] +
                                   U[1:m - 1:2, 3::2] -
                                   F[1:m - 1:2, 2:n - 1:2] * h * h) / (4.0)
        U[2:m - 1:2, 1:n - 1:2] = (U[1:m - 2:2, 1:n - 1:2] +
                                   U[3::2, 1:n - 1:2] +
                                   U[2:m - 1:2, 0:n - 2:2] +
                                   U[2:m - 1:2, 2::2] -
                                   F[2:m - 1:2, 1:n - 1:2] * h * h) / (4.0)
    else:
        # black
        U[1:m - 1:2, 1:n - 1:2] = (U[0:m - 2:2, 1:n - 1:2] +
                                   U[2::2, 1:n - 1:2] +
                                   U[1:m - 1:2, 0:n - 2:2] +
                                   U[1:m - 1:2, 2::2] -
                                   F[1:m - 1:2, 1:n - 1:2] * h * h) / (4.0)
        U[2:m - 1:2, 2:n - 1:2] = (U[1:m - 2:2, 2:n - 1:2] +
                                   U[3::2, 2:n - 1:2] +
                                   U[2:m - 1:2, 1:n - 2:2] +
                                   U[2:m - 1:2, 3::2] -
                                   F[2:m - 1:2, 2:n - 1:2] * h * h) / (4.0)
# ----------------

# --- 3D Fall ---


def sweep_3D(color, F, U, h):
    """
    Does the sweeps
    @param color 1 = red 0 for black
    @param h is distance between grid points
    """

    m, n, o = F.shape

    for k in range(1, o - 1):
        for j in range(1, n - 1):
            for i in range(1, m - 1):
                if (i + j + k) % 2 == color:
                    U[i, j, k] = (U[i - 1, j, k] +
                                  U[i + 1, j, k] +
                                  U[i, j - 1, k] +
                                  U[i, j + 1, k] +
                                  U[i, j, k - 1] +
                                  U[i, j, k + 1] -
                                  F[i, j, k]) / 6.0
